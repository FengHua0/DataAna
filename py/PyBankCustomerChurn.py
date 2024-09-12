from dataset_process import load_data, split_data

from py.create_feature import feature_create
from py.data_train import data_train
from py.select_feature import select_feature


def main():
    # 数据预处理
    data_path = '../data/Overfitting_data/data.csv'
    data = load_data(data_path)
    split_data(data)

    processed_data_path = '../data/processed_data.csv'
    customers_data = load_data(processed_data_path)
    # 数据分析可视化
    feature_create(customers_data)

    df = customers_data.drop(labels=['CLIENTNUM', 'Total_Relationship_Count', 'Contacts_Count_12_mon'], axis=1)
    df = df[df.columns[:-7]]

    df_gdbt,label = select_feature(df)

    data_train(df_gdbt,label )


if __name__ == '__main__':
    main()
