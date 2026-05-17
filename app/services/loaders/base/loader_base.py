from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Any

class DocumentLoaderBase(ABC):
    @abstractmethod
    def load(self,filename: str, file: BinaryIO, metadata: Dict[str, Any] = None) -> str:
        """
        Menerima file stream dan mengembalikan teks mentah hasil ekstraksi.
        """
        pass