"""Paired delayed recall, with deterministic family-level split assignment."""
import hashlib
import torch

BOS, RED, BLUE, QUERY, EOS, UNUSED, ANSWER_RED, ANSWER_BLUE = range(8)
TAG0, TAGS, NOISE0, NOISES = 8, 4, 12, 8
VOCAB, GROUP = NOISE0 + NOISES, 2 * TAGS
SPLITS = {'train': range(14), 'probe_train': range(14, 16),
          'validation': range(16, 18), 'test': range(18, 20)}


def family_bucket(noise):
    return int.from_bytes(hashlib.sha256(bytes(noise)).digest()[:8], 'big') % 20


def vocab_size(num_cues=1):
    return VOCAB if num_cues == 1 else VOCAB + num_cues


def anchor_index(num_cues=1):
    return num_cues + 2  # BOS, cues, nuisance tag, first distractor


def batch(n, delay, generator, split='train', num_cues=1):
    """BOS CUE TAG D_1 ... D_L QUERY ANSWER EOS; no padding.

    Each family contains all 2 cue x 4 nuisance-tag combinations with identical
    distractors. Entire families belong to one split, independently of cue/tag.
    All examples in a batch share a delay so official unmodified losses work.
    """
    if n < GROUP or n % GROUP or delay < 4 or num_cues < 1:
        raise ValueError('Use batches divisible by eight and delays >= 4.')
    rows, keys, seen = [], [], set()
    attempts = 0
    while len(rows) < n:
        attempts += 1
        if attempts > n * 1000:
            raise RuntimeError('Requested too many unique noise families for this split/delay.')
        noise = torch.randint(NOISE0, VOCAB, (delay,), generator=generator).tolist()
        key = bytes(noise)
        if family_bucket(noise) not in SPLITS[split] or key in seen:
            continue
        seen.add(key)
        keys.append(hashlib.sha256(key).hexdigest())
        # Preserve the original one-cue random stream exactly. In the multi-cue
        # task, the future query index is independent of every stored cue.
        query = int(torch.randint(num_cues, (), generator=generator)) if num_cues > 1 else 0
        bits = torch.randint(2, (num_cues,), generator=generator).tolist() if num_cues > 1 else [0]
        for cue in range(2):
            for tag in range(TAGS):
                memory = list(bits)
                memory[query] = cue
                query_token = QUERY if num_cues == 1 else VOCAB + query
                rows.append([BOS, *[RED+b for b in memory], TAG0+tag,
                             *noise, query_token, ANSWER_RED+cue, EOS])
    return torch.tensor(rows, dtype=torch.long), keys


def labels(seq, num_cues=1):
    query_index = torch.zeros(len(seq), dtype=torch.long) if num_cues == 1 else seq[:, -3]-VOCAB
    selected = seq[torch.arange(len(seq)), 1+query_index]-RED
    result = {'cue': selected.numpy(),
              'nuisance': (seq[:, num_cues+1]-TAG0).numpy()}
    if num_cues > 1:
        result['query_index'] = query_index.numpy()
        result.update({f'bit_{i}': (seq[:, 1+i]-RED).numpy() for i in range(num_cues)})
    return result
