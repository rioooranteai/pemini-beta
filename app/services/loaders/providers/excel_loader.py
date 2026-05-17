import pandas as pd
import zipfile
import os
import shutil
import re
from typing import BinaryIO, Dict, Any
from app.services.loaders.base.loader_base import DocumentLoaderBase
from app.services.loaders.models import ExcelLoadResult
from app.core.config import get_settings


class SmartExcelLoaderProvider(DocumentLoaderBase):
    def load(self, filename: str, file: BinaryIO, metadata: Dict[str, Any] = None) -> ExcelLoadResult:
        if metadata is None: metadata = {}
        settings = get_settings()

        # 1. SIAPKAN FOLDER GAMBAR
        safe_filename = os.path.splitext(filename)[0].replace(" ", "_")
        doc_image_dir = os.path.join(settings.images_path, safe_filename)
        os.makedirs(doc_image_dir, exist_ok=True)
        image_paths = {'global': []}

        # 2. EKSTRAK GAMBAR VIA ZIP
        try:
            file.seek(0)
            with zipfile.ZipFile(file) as archive:
                media_files = [f for f in archive.namelist() if f.startswith('xl/media/')]
                for media_file in media_files:
                    filename_only = os.path.basename(media_file)
                    target_path = os.path.join(doc_image_dir, filename_only)
                    with archive.open(media_file) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    image_paths['global'].append(target_path)
        except zipfile.BadZipFile:
            pass  # Lanjutkan saja jika gagal (misal file CSV)

        # 3. BACA TEKS VIA PANDAS
        file.seek(0)
        all_sheets = pd.read_excel(file, sheet_name=None, header=None)
        text_content, dataframes = {}, {}

        for sheet_name, df in all_sheets.items():
            df = df.dropna(how='all').dropna(axis=1, how='all')
            extracted_data = {}
            judul_dokumen = ""

            # (Gunakan logika "Sliding Window" yang sama dengan yang kita bahas sebelumnya
            #  untuk mengambil pasangan Key-Value dari tiap baris)
            for index, row in df.iterrows():
                valid_cells = [str(cell).strip() for cell in row if pd.notna(cell) and str(cell).strip()]
                if len(valid_cells) == 1:
                    if not judul_dokumen:
                        judul_dokumen = valid_cells[0]
                    else:
                        extracted_data[f"Catatan_{index}"] = valid_cells[0]
                elif len(valid_cells) >= 2:
                    i = 0
                    while i < len(valid_cells) - 1:
                        if "Unnamed" not in valid_cells[i]:
                            extracted_data[valid_cells[i]] = valid_cells[i + 1]
                        i += 2
                    if len(valid_cells) % 2 != 0:
                        extracted_data[f"Info_{index}"] = valid_cells[-1]

            # 4. FORMAT OUTPUT
            # Teks untuk VDB
            rag_text = f"Dokumen: {judul_dokumen}\nSheet: {sheet_name}\n"
            for k, v in extracted_data.items(): rag_text += f"- {k}: {v}\n"
            text_content[sheet_name] = rag_text

            # DataFrame untuk SQL
            sql_df = pd.DataFrame([extracted_data])
            sql_df.insert(0, 'Sheet_Name', sheet_name)
            sql_df.insert(0, 'Judul_Dokumen', judul_dokumen)

            clean_cols = []
            for col in sql_df.columns:
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
                if not clean_name or clean_name[0].isdigit(): clean_name = "Col_" + clean_name
                clean_cols.append(clean_name)
            sql_df.columns = clean_cols
            dataframes[sheet_name] = sql_df

        return ExcelLoadResult(
            text_content=text_content, dataframes=dataframes,
            sheet_names=list(all_sheets.keys()), metadata=metadata, image_paths=image_paths
        )