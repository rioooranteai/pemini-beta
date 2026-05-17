from app.services.loaders.base.loader_base import DocumentLoaderBase
from app.services.loaders.providers.excel_loader import ExcelLoaderProvider


class LoaderFactory:

    @staticmethod
    def create() -> DocumentLoaderBase:
        return ExcelLoaderProvider()
