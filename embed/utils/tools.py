def pad_seq(seq, max_len, pad_token=0):
    seq += [pad_token for i in range(max_len - len(seq))]
    return seq