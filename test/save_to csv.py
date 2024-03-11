import csv


def save_to_csv(data, path):
    with open(path, 'w+', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerows(data)


if __name__ == '__main__':
    data = [['a', 'b', 'c', 1, 2],
            [1, 2, 3, ],
            [1, 2, 3, 4, ]]
    save_to_csv(data, 'test.csv')
