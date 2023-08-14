import pandas as pd


def score(ac, check):
    ac = pd.read_csv(ac, header=None)
    check = pd.read_csv(check, header=None)

    ac = ac.astype(str)
    check = check.astype(str)

    comparison_df = (ac != check)

    ac = ac[comparison_df]
    check = check[comparison_df]

    ac.dropna(axis=0, how='all', inplace=True)
    ac.dropna(axis=1, how='all', inplace=True)
    check.dropna(axis=0, how='all', inplace=True)
    check.dropna(axis=1, how='all', inplace=True)

    # print("==================== ac error ===================")
    # print(ac)
    # print("\n================== check error ==================")
    # print(check)
    # print("\n===================== score =====================")

    num_different_cells = comparison_df.sum().sum() / (comparison_df.shape[0] * comparison_df.shape[1])

    print("ac rate: ", round(1 - num_different_cells, 4))

    with open("page_2.txt", "a") as f:
        f.write(f"ac rate: {round(1 - num_different_cells, 4)}\n")


def tsc_to_csv():
    with open("ac.csv", 'r') as f:
        s = f.read()

    s = s.replace("\t", ",")

    with open("ac.csv", 'w') as f:
        f.write(s)
