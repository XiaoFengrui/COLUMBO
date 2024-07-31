import random
import pandas as pd

def is_number(s):
    """
    chech number in str
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def shuffle_dic(dicts):
    """
    shuffle dic
    """
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def read_payloads(path):
    payloads = []
    with open(path) as f:
        while True:
            line = f.readline().strip()
            if line:
                payloads.append(line+' ')
            else:
                break
    return payloads

def read_payloads_csv(path):
    # 读取TSV文件
    df = pd.read_csv(path, delimiter='\t', names=['payload', 'label'], skiprows=1)
    
    # 过滤label为1的行
    filtered_df = df[df['label'] == 1]
    
    # 提取payload列并加上空格
    payloads = (filtered_df['payload'] + ' ').tolist()
    labels = filtered_df['label'].astype(int).tolist()
    
    return payloads, labels