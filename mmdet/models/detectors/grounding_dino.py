# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 text_mapper_hidden_dim=1024,
                 text_mapper_type='mlp',
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        # Modified: Use comma and space for LLM natural language understanding
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.text_mapper_hidden_dim = text_mapper_hidden_dim
        self.text_mapper_type = text_mapper_type
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        
        # Modified: Support dynamic language_dim for custom LLM2Vec model
        if hasattr(self.language_model, 'language_dim'):
            lang_dim = self.language_model.language_dim
        else:
            lang_dim = self.language_model.language_backbone.body.language_dim

        # self.text_feat_map = nn.Linear(
        #     lang_dim,
        #     self.embed_dims,
        #     bias=True)
        text_mapper_type = str(self.text_mapper_type).lower()
        text_mapper_hidden_dim = self.text_mapper_hidden_dim
        if text_mapper_type == 'linear' or (
                text_mapper_hidden_dim is not None
                and text_mapper_hidden_dim <= 0):
            self.text_feat_map = nn.Linear(
                lang_dim,
                self.embed_dims,
                bias=True)
        elif text_mapper_type == 'mlp':
            self.text_feat_map = nn.Sequential(
                nn.Linear(lang_dim, text_mapper_hidden_dim),
                nn.LayerNorm(text_mapper_hidden_dim),
                nn.GELU(),
                nn.Linear(text_mapper_hidden_dim, self.embed_dims, bias=True)
            )
        else:
            raise ValueError(
                f'Unsupported text_mapper_type={self.text_mapper_type}. '
                'Expected one of: "mlp", "linear".')

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        # nn.init.constant_(self.text_feat_map.bias.data, 0)
        # nn.init.xavier_uniform_(self.text_feat_map.weight.data)
        if isinstance(self.text_feat_map, nn.Sequential):
            # 遍历 Sequential 里的所有子模块
            for m in self.text_feat_map.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    # 检查是否存在 bias (因为我们第一层设了 bias=False)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)
        else:
            # 兼容老版本的单层写法
            nn.init.xavier_uniform_(self.text_feat_map.weight.data)
            nn.init.constant_(self.text_feat_map.bias.data, 0)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt',
                return_offsets_mapping=True)  # Used by get_positive_map.
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt',
                return_offsets_mapping=True)  # Used by get_positive_map.
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    # =========================================================================
    # 【核心修改 1】：彻底重写 get_positive_map，加入 caption_string 参数
    # =========================================================================
    @staticmethod
    def _get_char_to_token(tokenized, char_pos: int) -> Optional[int]:
        """Compatibility wrapper for tokenizer.char_to_token()."""
        if char_pos < 0 or not hasattr(tokenized, 'char_to_token'):
            return None

        try:
            pos = tokenized.char_to_token(0, char_pos)
            if pos is not None:
                return pos
        except TypeError:
            pass
        except Exception:
            return None

        try:
            return tokenized.char_to_token(char_pos)
        except Exception:
            return None

    @staticmethod
    def _get_offset_mapping(tokenized) -> Optional[list]:
        offsets = tokenized.get('offset_mapping', None)
        if offsets is None:
            return None
        if torch.is_tensor(offsets):
            offsets = offsets.detach().cpu().tolist()
        if len(offsets) == 0:
            return []

        first_elem = offsets[0]
        if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0 \
                and isinstance(first_elem[0], (list, tuple)):
            offsets = first_elem

        return [(int(start), int(end)) for start, end in offsets]

    @staticmethod
    def _to_prompt_entity_name(name: str) -> str:
        """Convert raw class name to prompt-friendly text.

        Keep disambiguation words inside parentheses (e.g. `bat_(animal)` ->
        `bat animal`) to reduce ambiguity in LVIS.
        """
        name = name.replace('_', ' ')
        name = name.replace('(', ' ').replace(')', ' ')
        name = re.sub(r'\s+', ' ', name)
        return name.strip()

    def _split_entities_by_token_budget(self, prompt_entities: list,
                                        display_entities: list) -> list:
        """Split entity list into chunks that fit language_model.max_tokens."""
        max_tokens = self.language_model.max_tokens
        chunks_ = []
        cur_prompt = []
        cur_display = []

        for prompt_name, display_name in zip(prompt_entities, display_entities):
            test_prompt = cur_prompt + [prompt_name]
            caption_test, _ = self.to_plain_text_prompts(test_prompt)
            tokenized_test = self.language_model.tokenizer([caption_test],
                                                           return_tensors='pt')
            token_len = tokenized_test.input_ids.shape[1]
            if len(cur_prompt) > 0 and token_len > max_tokens:
                chunks_.append((cur_prompt, cur_display))
                cur_prompt = [prompt_name]
                cur_display = [display_name]
            else:
                cur_prompt = test_prompt
                cur_display = cur_display + [display_name]

        if len(cur_prompt) > 0:
            chunks_.append((cur_prompt, cur_display))
        return chunks_

    def _force_single_class_alignment(self, tokenized, tokens_positive,
                                      caption_string):
        """Force a non-empty map for a single class when all matching fails."""
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        positive_map = torch.zeros((1, max_text_len), dtype=torch.float)

        offset_mapping = self._get_offset_mapping(tokenized)
        if offset_mapping is not None and len(tokens_positive) > 0 and \
                len(tokens_positive[0]) > 0:
            char_start, char_end = tokens_positive[0][0]
            char_start = max(int(char_start), 0)
            char_end = max(int(char_end), char_start)
            while char_start < char_end and caption_string[char_start].isspace():
                char_start += 1
            while char_end > char_start and caption_string[char_end - 1].isspace():
                char_end -= 1
            for tok_ind, (tok_start, tok_end) in enumerate(
                    offset_mapping[:max_text_len]):
                if tok_end <= tok_start:
                    continue
                if tok_start < char_end and tok_end > char_start:
                    positive_map[0, tok_ind] = 1.0

        # Fallback: use the first valid token span.
        if positive_map[0].sum() == 0 and offset_mapping is not None:
            for tok_ind, (tok_start, tok_end) in enumerate(
                    offset_mapping[:max_text_len]):
                if tok_end > tok_start:
                    positive_map[0, tok_ind] = 1.0
                    break

        # Last fallback: use the first non-pad token.
        if positive_map[0].sum() == 0 and 'attention_mask' in tokenized:
            attention_mask = tokenized['attention_mask'][0]
            if torch.is_tensor(attention_mask):
                attention_mask = attention_mask.detach().cpu().tolist()
            for tok_ind, mask_v in enumerate(attention_mask[:max_text_len]):
                if mask_v:
                    positive_map[0, tok_ind] = 1.0
                    break

        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def _legacy_get_positive_map(self,
                                 tokenized,
                                 tokens_positive,
                                 caption_string=None):
        """Create token-level positive map from character-level spans."""
        max_text_len = self.bbox_head.cls_branches[self.decoder.num_layers].max_text_len
        positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)
        
        # 获取全文的 input_ids
        input_ids = tokenized['input_ids'][0].tolist()
        tokenizer = self.language_model.tokenizer
        
        for j, tok_pos in enumerate(tokens_positive): 
            for (char_start, char_end) in tok_pos:
                # 必须有 original string 才能截取子串，否则跳过
                if caption_string is None:
                    continue
                    
                target_word = caption_string[char_start:char_end]
                
                # 处理 Tiktoken 前置空格特性的多种匹配模式
                word_tokens_a = tokenizer(target_word, add_special_tokens=False).input_ids
                word_tokens_b = tokenizer(" " + target_word, add_special_tokens=False).input_ids
                
                found = False
                for search_ids in [word_tokens_b, word_tokens_a]:
                    if found or len(search_ids) == 0:
                        continue
                        
                    n = len(search_ids)
                    for k in range(len(input_ids) - n + 1):
                        if input_ids[k : k + n] == search_ids:
                            # 找到了匹配的子串，写入 positive_map
                            actual_len = min(k + n, max_text_len)
                            if k < max_text_len:
                                positive_map[j, k : actual_len] = 1.0
                            found = True
                            break

        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
            
        return positive_map_label_to_token, positive_map

    def get_positive_map(self, tokenized, tokens_positive, caption_string=None):
        """Create token-level positive map from character-level spans."""
        # Keep backward-compatible behavior for non-chunked settings
        # (e.g., COCO eval in this project), and use robust alignment for
        # chunked inference (e.g., LVIS with many categories).
        chunked_size = self.test_cfg.get('chunked_size', -1)
        if self.training or chunked_size <= 0:
            return self._legacy_get_positive_map(tokenized, tokens_positive,
                                                 caption_string)

        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        positive_map = torch.zeros(
            (len(tokens_positive), max_text_len), dtype=torch.float)

        offset_mapping = self._get_offset_mapping(tokenized)
        tokenizer = self.language_model.tokenizer
        input_ids = tokenized['input_ids'][0]
        if torch.is_tensor(input_ids):
            input_ids = input_ids.detach().cpu().tolist()

        for j, tok_pos in enumerate(tokens_positive):
            for (char_start, char_end) in tok_pos:
                if caption_string is None:
                    continue

                char_start = max(int(char_start), 0)
                char_end = max(int(char_end), char_start)
                # Trim whitespace boundaries to avoid mapping spans that
                # end/start on separator spaces.
                while char_start < char_end and caption_string[char_start].isspace():
                    char_start += 1
                while char_end > char_start and caption_string[char_end - 1].isspace():
                    char_end -= 1
                if char_end <= char_start:
                    continue

                matched_token_inds = []

                # Preferred path: offset overlap on [char_start, char_end).
                if offset_mapping is not None:
                    valid_len = min(len(offset_mapping), max_text_len)
                    for tok_ind in range(valid_len):
                        tok_start, tok_end = offset_mapping[tok_ind]
                        if tok_end <= tok_start:
                            continue
                        if tok_start < char_end and tok_end > char_start:
                            matched_token_inds.append(tok_ind)

                # Fallback path: char_to_token compatibility for tokenizer
                # variants that do not expose usable offset mappings.
                if len(matched_token_inds) == 0:
                    beg_pos = self._get_char_to_token(tokenized, char_start)
                    end_pos = self._get_char_to_token(tokenized, char_end - 1)

                    if beg_pos is None:
                        for delta in (1, 2):
                            beg_pos = self._get_char_to_token(
                                tokenized, char_start + delta)
                            if beg_pos is not None:
                                break
                    if end_pos is None:
                        for delta in (1, 2):
                            end_pos = self._get_char_to_token(
                                tokenized, char_end - 1 - delta)
                            if end_pos is not None:
                                break

                    if beg_pos is not None and end_pos is not None:
                        beg_pos = max(0, beg_pos)
                        end_pos = min(end_pos, max_text_len - 1)
                        if beg_pos <= end_pos:
                            matched_token_inds = list(
                                range(beg_pos, end_pos + 1))

                # Last fallback: token-id search, but pick the match nearest
                # to the target char span to avoid global-first-match errors.
                if len(matched_token_inds) == 0:
                    target_word = caption_string[char_start:char_end]
                    if len(target_word) > 0:
                        candidate_token_ids = []
                        token_ids_with_space = tokenizer(
                            ' ' + target_word,
                            add_special_tokens=False).input_ids
                        token_ids_no_space = tokenizer(
                            target_word,
                            add_special_tokens=False).input_ids
                        if len(token_ids_with_space) > 0:
                            candidate_token_ids.append(token_ids_with_space)
                        if len(token_ids_no_space) > 0 and \
                                token_ids_no_space != token_ids_with_space:
                            candidate_token_ids.append(token_ids_no_space)

                        best_match = None
                        for token_ids in candidate_token_ids:
                            token_num = len(token_ids)
                            for k in range(len(input_ids) - token_num + 1):
                                if input_ids[k:k + token_num] != token_ids:
                                    continue
                                if k >= max_text_len:
                                    continue
                                end_k = min(k + token_num, max_text_len)
                                if offset_mapping is not None and \
                                        k < len(offset_mapping):
                                    distance = abs(
                                        offset_mapping[k][0] - char_start)
                                else:
                                    distance = k
                                # Prefer nearer match first; if tied, prefer
                                # longer token spans.
                                match = (distance, -token_num, k, end_k)
                                if best_match is None or match < best_match:
                                    best_match = match

                        if best_match is not None:
                            matched_token_inds = list(
                                range(best_match[2], best_match[3]))

                if len(matched_token_inds) > 0:
                    positive_map[j, matched_token_inds] = 1.0

        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)

        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption."""
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt',
                    return_offsets_mapping=True)
                
                # =================================================================
                # 【核心修改 2】：向 get_positive_map 传入 original_caption 
                # =================================================================
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive, original_caption)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            
            # =================================================================
            # 【核心修改 3】：向 get_positive_map 传入 caption_string 
            # =================================================================
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive, caption_string)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        raw_caption = list(original_caption)
        display_entities = [clean_label_name(i) for i in raw_caption]
        if enhanced_text_prompts is None:
            prompt_entities = [
                self._to_prompt_entity_name(i) for i in raw_caption
            ]
        else:
            # Keep enhanced prompt key matching behavior consistent with
            # previous logic (keys are usually based on cleaned class names).
            prompt_entities = display_entities

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        def process_chunk(prompt_chunk: list, display_chunk: list) -> None:
            if len(prompt_chunk) == 0:
                return

            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    display_chunk, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    prompt_chunk)

            tokenized = self.language_model.tokenizer(
                [caption_string], return_tensors='pt', return_offsets_mapping=True)
            token_len = tokenized.input_ids.shape[1]
            if token_len > self.language_model.max_tokens and \
                    len(prompt_chunk) > 1:
                mid = max(1, len(prompt_chunk) // 2)
                process_chunk(prompt_chunk[:mid], display_chunk[:mid])
                process_chunk(prompt_chunk[mid:], display_chunk[mid:])
                return
            if token_len > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')

            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive, caption_string)
            unaligned_inds = [
                idx for idx in range(positive_map.shape[0])
                if positive_map[idx].sum() == 0
            ]
            if len(unaligned_inds) > 0 and len(prompt_chunk) > 1:
                mid = max(1, len(prompt_chunk) // 2)
                process_chunk(prompt_chunk[:mid], display_chunk[:mid])
                process_chunk(prompt_chunk[mid:], display_chunk[mid:])
                return
            if len(unaligned_inds) > 0 and len(prompt_chunk) == 1:
                positive_map_label_to_token, positive_map = \
                    self._force_single_class_alignment(
                        tokenized, tokens_positive, caption_string)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(display_chunk)

        if enhanced_text_prompts is not None:
            # For enhanced prompts, keep original fixed-size chunking behavior
            # as the text template can significantly change token lengths.
            if chunked_size <= 0:
                chunked_size = max(1, len(prompt_entities))
            initial_prompt_chunks = chunks(prompt_entities, chunked_size)
            initial_display_chunks = chunks(display_entities, chunked_size)
            initial_chunks = list(zip(initial_prompt_chunks,
                                      initial_display_chunks))
        else:
            initial_chunks = self._split_entities_by_token_budget(
                prompt_entities, display_entities)

        for prompt_chunk, display_chunk in initial_chunks:
            process_chunk(prompt_chunk, display_chunk)

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

            # =================================================================
            # 【核心修改 4】：向 get_positive_map 传入 caption_string 
            # =================================================================

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt',
                    return_offsets_mapping=True)
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                # =================================================================
                # 【核心修改 5】：向 get_positive_map 传入 text_prompt
                # =================================================================
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive, text_prompt)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # =================================================================
                    # 【核心修改 6】：向 get_positive_map 传入 caption_string
                    # =================================================================
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive, caption_string)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # =================================================================
                    # 【核心修改 7】：向 get_positive_map 传入 caption_string
                    # =================================================================
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive, caption_string)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
