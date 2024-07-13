import os.path

import pdfplumber
from PIL import Image


def export_imgs(pdf_path, img_dir):
    with pdfplumber.open(pdf_path) as pdf:
        # print(len(pdf.images))
        for page in pdf.pages:
            for i, img_obj in enumerate(page.images):
                name = img_obj['name']
                page_number = img_obj['page_number']
                print('页数：', page_number, '  图片名：', name)

                print(img_obj)

                path = os.path.join(img_dir, f"{page_number}-{i + 1}-{name}.jpg")
                open(path, 'wb').write(img_obj['stream'].get_data())

                # print(img_obj['stream'].get_data())
                # print(img_obj['stream'].get_rawdata())
                # print(dir(img_obj))
                # print(dir(img_obj['stream']))
                width = img_obj['width']
                height = img_obj['height']
                srcsize = img_obj['srcsize']
                # img = Image.frombytes("RGB", srcsize, img_obj['stream'].get_data())
                # img.show()


def test_img(page_num):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        print(page.images)
        for obj in page.objects:
            print(obj, '=================================')
            print(page.objects[obj])


if __name__ == '__main__':
    pdf_path = r"F:\Download\Matlab\BDA\DA.pdf"
    img_dir = r"F:\Download\Matlab\BDA\DA"

    export_imgs(pdf_path, img_dir)
    # test_img(3)
