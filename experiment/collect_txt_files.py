import os 


def main(path):
    with open('./collect_data.txt', 'w', encoding='utf-8') as f:
        txt_files = [f for f in os.listdir(path) if f.find('.txt') != -1]
        for id, txt_name in enumerate(txt_files):
            txt_file_path = os.path.join(path, txt_name)
            with open(txt_file_path, 'r', encoding='utf-8') as g:
                a = g.readlines()
                if a[-1][-1:] != '\n':
                    f.writelines(a)
                    f.write("\n")
                else:
                    f.writelines(a)
            if id != len(txt_files) - 1:
                f.write("\n")
                

if __name__ == '__main__':
    path = './conversation'
    main(path)
    