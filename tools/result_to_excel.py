# -*- coding: utf-8 -*-

import pandas as pd
import xlsxwriter
import os
import json
from vision_demo import upload_file
import time
from tqdm import tqdm


def result_to_excel():
    speaker = "molinger-moyin"
    date = "20230815"

    test_dir = "/data1/xiepengyuan/workspace/audio/wetts/test"
    excel_path = os.path.join(test_dir, "results", date, "excel", f"{speaker}.xlsx")
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}_s3.json")
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    audio_indexes = [item["index"] for item in items]
    audio_urls = [item["audio_url"] for item in items]
    texts = [item["text"] for item in items]

    # 创建示例数据框
    df = pd.DataFrame({
        'audio_indexes': audio_indexes,
        'texts': texts,
        'audio_urls': audio_urls
    })

    # 创建Excel文件
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet()

    # 设置超链接格式
    link_format = workbook.add_format({'color': 'blue', 'underline': 1})

    # 写入数据框
    worksheet.write('A1', '序号')
    worksheet.write('B1', '文本')
    worksheet.write('C1', '音频')
    for i, row in df.iterrows():
        worksheet.write(i + 1, 0, row["audio_indexes"])
        worksheet.write(i + 1, 1, row["texts"])
        worksheet.write_url(i + 1, 2, row['audio_urls'], link_format, string=row['audio_indexes'])

    # 保存Excel文件
    workbook.close()


if __name__ == '__main__':
    result_to_excel()
