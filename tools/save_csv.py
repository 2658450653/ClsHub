import pandas as pd

if __name__ == '__main__':
    text = []
    with open('../output/rocks_extend/t2t_vit_10/log.txt', 'r') as f:
        while True:
            r = f.readline()
            if r == '' or r == '\n':
                break
            text.append(eval(r))
            print(r)
    df = pd.DataFrame(text, columns=list(text[0].keys()))
    print(df.head(10))
    df.to_csv('../output/rocks_extend/t2t_vit_10/'
              + 'result.csv')
