from typing import Protocol, BinaryIO

class File(Protocol):
    filename: str
    stream: BinaryIO

class FileWrapper:
    def __init__(self, buffer, file: File):
        self.stream = buffer
        self.filename = file.filename
