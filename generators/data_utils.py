# here put the import lib
import copy
import numpy as np


def random_neq(l, r, s=[]):    # 在l-r之间随机采样一个数，这个数不能在列表s中
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def truncate_padding(inter, domain_mask, max_len, item_nums, domain_offsets):
    seq = np.zeros([max_len], dtype=np.int32)
    pos = np.zeros([max_len], dtype=np.int32)
    neg = np.zeros([max_len], dtype=np.int32)
    processed_domain_mask = np.ones([max_len], dtype=np.int32) * -1 # -1で初期化

    if len(inter) > 1:
        # non_negにはサンプリングから除外するグローバルIDのセット
        non_neg = set(inter)
        
        # 最後のアイテムとそのドメインIDを初期値とする
        nxt = inter[-1]
        nxt_domain = domain_mask[-1]
        
        idx = max_len - 1
        
        # reversed()を使ってシーケンスとドメインマスクを最後から2番目から逆順にループ
        for i, d in reversed(list(zip(inter[:-1], domain_mask[:-1]))):
            if idx < 0:
                break
            
            seq[idx] = i
            pos[idx] = nxt
            processed_domain_mask[idx] = nxt_domain
            
            # --- ネガティブサンプリング (Nドメイン対応) ---
            # d (現在のアイテムのドメイン) と同じドメインからサンプリング
            # ここではローカルIDの範囲でサンプリングし、後でオフセットを足す
            domain_item_num = item_nums[d]
            sampled_neg_local = random_neq(1, domain_item_num + 1, []) # ローカルIDでサンプリング
            
            # グローバルIDに変換
            neg[idx] = sampled_neg_local + domain_offsets[d]

            # 次の反復のために、nxtとnxt_domainを現在のものに更新
            nxt = i
            nxt_domain = d
            idx -= 1
            
    # ポジションの計算
    true_len = len(inter)
    positions = np.zeros([max_len], dtype=np.int32)
    if true_len > 0:
        # 実際に存在するシーケンスの長さを計算
        seq_len = min(true_len, max_len)
        # 右詰めでポジションを設定
        positions[-seq_len:] = np.arange(1, seq_len + 1)

    return seq, pos, neg, positions, processed_domain_mask