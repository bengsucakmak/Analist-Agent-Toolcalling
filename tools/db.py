# tools/db.py
import sqlite3
import json
import logging
from typing import List
from langchain_core.tools import tool, BaseTool

log = logging.getLogger("db_tools")

# Sınıf yerine, araçları üreten tek bir fonksiyon kullanıyoruz.
# Bu, "self" ile ilgili tüm sorunları ortadan kaldırır.
def create_database_tools(db_path: str) -> List[BaseTool]:
    """
    Veritabanı yolunu alıp, bu yola göre yapılandırılmış,
    @tool ile dekore edilmiş bir araç listesi döndüren bir fabrika fonksiyonu.
    """
    if not db_path:
        raise ValueError("Database path cannot be empty.")
    
    # --- Yardımcı Fonksiyon (Kapsülleme için) ---
    def _connect_readonly() -> sqlite3.Connection:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # --- @tool ile Dekore Edilmiş Gerçek Araçlar ---
    # Bu fonksiyonlar artık bir sınıf içinde değil, başka bir fonksiyon içinde.
    # Bu yüzden 'self' argümanı almazlar ve Pydantic hatası oluşmaz.

    @tool
    def run_sql(query: str) -> str:
        """
        Verilen SQL sorgusunu veritabanında çalıştırır ve sonucu JSON formatında bir dize olarak döndürür.
        Veri hakkında kullanıcı sorularını yanıtlamak için bu aracı kullanın. Girdi geçerli bir SQLite sorgusu olmalıdır.
        """
        try:
            conn = _connect_readonly()
            cur = conn.execute(query)
            colnames = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(colnames, row)) for row in cur.fetchall()]
            result = {"columns": colnames, "rows": rows}
            return json.dumps(result, indent=2, ensure_ascii=False)
        except sqlite3.Error as e:
            log.error(f"SQL Hatası: {e}\nSorgu: {query}")
            return f"Veritabanı hatası: {e}"
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    @tool
    def get_schema() -> str:
        """
        Veritabanının şemasını (tablo adları ve sütunlar) döndürür.
        SQL yazmadan önce hangi tabloları ve sütunları sorgulamanız gerektiğini anlamak için bu aracı kullanın.
        """
        try:
            conn = _connect_readonly()
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            tables = [r["name"] for r in cur.fetchall()]
            schema_parts = []
            for table_name in tables:
                cur = conn.execute(f"PRAGMA table_info({table_name});")
                columns = [f"{row['name']}:{row['type']}" for row in cur.fetchall()]
                schema_parts.append(f"Table '{table_name}' — columns: [{', '.join(columns)}]")
            return "\n".join(schema_parts)
        except sqlite3.Error as e:
            log.error(f"Şema alınırken hata: {e}")
            return f"Veritabanı hatası: {e}"
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    # Fonksiyon, ürettiği araçları bir liste olarak döndürür.
    return [run_sql, get_schema]