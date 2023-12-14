#Clear
import os
import glob
import shutil
from PIL import Image
import argparse

files = glob.glob('input/*/*/*.*')
for f in files:
  os.remove(f)

files = glob.glob('results/*/*/*.*')
for f in files:
  os.remove(f)

files = glob.glob('final/*/*.*')
for f in files:
  os.remove(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Full inference script")

    parser.add_argument(
        "--category",
        type=str,
        default="upper_body",
        help="cloth category for virtual try on",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="human image path for virtual try on",
    )

    parser.add_argument(
        "--cloth",
        type=str,
        default="",
        help="cloth image path for virtual try on",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

args = parse_args()

def get_image_filename(image_path):
    image_filename = os.path.basename(image_path)
    return image_filename

category = args.category
if category == 'upper_body':
   var_category = '0'
elif category == 'lower_body':
   var_category = '1'
else:
   var_category = '2'
image_path = args.image
cloth_path = args.cloth
image_filename = get_image_filename(image_path)
cloth_filename = get_image_filename(cloth_path)
print("image_filename:", image_filename)

def resize_with_pad(im, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    print("im_ratio:", im_ratio)
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')

#Add pairs
def write_row(file_, *columns):
    print(*columns, sep='\t', end='\n', file=file_)

def write_three_columns(file_path, column1, column2, column3):
    with open(file_path, 'a') as file:
        line = f"{column1}\t{column2}\t{column3}\n"
        file.write(line)

upper = open('input/upper_body/test_pairs_unpaired.txt', 'w')
lower = open('input/lower_body/test_pairs_unpaired.txt', 'w')
dresses = open('input/dresses/test_pairs_unpaired.txt', 'w')
all = open('input/test_pairs_paired.txt', 'w')

test_pairs_path = 'images/test_pairs.txt'

with open(test_pairs_path, 'w') as file:
    file.write('')

write_three_columns(test_pairs_path, image_filename, cloth_filename, var_category)

with open('images/test_pairs.txt', "r") as file:
    data = file.readlines()
    print("data:", data)
    for line in data:
        word = line.split()
        if len(word) >= 3:  # Check if word has at least 3 elements
            org_path = 'images/humans/' + word[0]
            # Rest of your code for writing and resizing images
        else:
            print("Invalid line format:", line)
        #print("word:", word)
        #org_path = 'images/humans/' + word[0]
        print("org_path:", org_path)
        if(word[2] == '0'):
          write_row(upper,'0'+word[0],word[1])
          write_row(all,'0'+word[0],word[1],word[2])
          res_path = 'input/upper_body/images/0' + word[0]
        elif(word[2] == '1'):
          write_row(lower,'1'+word[0],word[1])
          write_row(all,'1'+word[0],word[1],word[2])
          res_path = 'input/lower_body/images/1' + word[0]
        elif(word[2] == '2'):
          write_row(dresses,'2'+word[0],word[1])
          write_row(all,'2'+word[0],word[1],word[2])
          res_path = 'input/dresses/images/2' + word[0]
        image = Image.open(org_path)
        new = resize_with_pad(image,384,512)
        new.save(res_path)

upper.close()
lower.close()
dresses.close()
all.close()