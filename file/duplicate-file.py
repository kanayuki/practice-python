import os

from rich.style import Style
from rich.table import Table

from rich.console import Console

console = Console()


def duplicate_file(source_dir, dest_dir):
    table = Table()

    source_files = os.listdir(source_dir)
    dest_files = os.listdir(dest_dir)
    table.add_column(source_dir + f'  ({len(source_files)}个文件)')
    table.add_column('文件大小')
    table.add_column(dest_dir + f'  ({len(dest_files)}个文件)')

    for f in sorted(set(source_files + dest_files)):
        c1, c2, c3 = [''] * 3
        source_size, dest_size = 0, 0
        style = ''
        if f in source_files:
            c1 = f
            source_size = os.path.getsize(os.path.join(source_dir, f))

        if f in dest_files:
            c3 = f
            dest_size = os.path.getsize(os.path.join(dest_dir, f))

        if source_size == dest_size:
            c2 = str(source_size)
        else:
            c2 = f'{source_size} | {dest_size}'
            style = Style(bgcolor='red')

        table.add_row(c1, c2, c3, style=style)

    console.print(table)


if __name__ == '__main__':
    duplicate_file(r'I:\Beauty', r'G:\Beauty')
