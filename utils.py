import os
from datetime import datetime


class EXIOStream:
    def __init__(self, dir_path: str, main_name: str = "main.log") -> None:
        self.dir = dir_path
        self.main_name = main_name
        os.makedirs(self.dir, exist_ok=True)
        self.fmain = open(os.path.join(self.dir, main_name), "a", encoding="utf-8")

    def cprint(self, *args, name=None, to_main=True) -> None:
        if to_main:
            if name is None:
                print(f"[{self.main_name}]", *args)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, file=self.fmain)
            self.fmain.flush()
        if name is not None and name != self.main_name:
            print(f"[{name}]", *args)
            with open(os.path.join(self.dir, name), "a", encoding="utf-8") as fout:
                print(*args, file=fout)
