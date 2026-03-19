import pandas as pd
import os

from hotel_ipa.constants import HOTEL_ORDER

INPUT_FILE = 'data/raw/Raw_dataset_for_JBR_paper.xlsx'
OUTPUT_FILE = 'data/processed/sorted_dataset.csv'

def load_data(file_path: str) -> pd.DataFrame:
    # 讀取 Excel 檔案
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    print(f"正在讀取資料: {file_path}...")
    return pd.read_excel(file_path)

def process_data(df: pd.DataFrame, custom_order: list) -> pd.DataFrame:
    # 處理資料：日期格式化與自訂排序
    # 建立副本以免影響原始資料
    df_processed = df.copy()

    # 日期格式化
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date']) # 確保是 datetime 物件
        df_processed['Date'] = df_processed['Date'].dt.strftime('%Y/%m/%d')
    
    # 設定類別型資料排序
    if 'Hotel Name' in df_processed.columns:
        df_processed['Hotel Name'] = pd.Categorical(
            df_processed['Hotel Name'], 
            categories=custom_order, 
            ordered=True
        )
    
    # 執行排序
    df_sorted = df_processed.sort_values(by=['Hotel Name', 'Date'])
    return df_sorted

def save_data(df: pd.DataFrame, file_path: str):
    # 儲存資料（依副檔名自動選擇 CSV 或 Excel）
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        print(f"排序完成，結果已儲存為: {file_path}")
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")

def main():
    try:
        df = load_data(INPUT_FILE)
        
        df_sorted = process_data(df, HOTEL_ORDER)

        save_data(df_sorted, OUTPUT_FILE)
        
    except Exception as e:
        print(f"程式執行中斷: {e}")

if __name__ == "__main__":
    main()