matrix_dimensions = 28
max_length_index = matrix_dimensions ** 2


# Extract formatted data from files indicated
def extract_data(images_file: str, labels_file: str) -> (list, list):
    data = []
    number = []
    classes = []

    with open(images_file) as file:

        lines = file.readlines()
        index = 0

        for line in lines:

            # Foreach character in line without last '\n'
            for character in line[:-1]:

                # Depends of character assign a number (0 to blank, 1 to plus, and 2 to others)
                if character == ' ':
                    number.append(0)
                elif character == '+':
                    number.append(1)
                else:
                    number.append(2)

                index += 1

                if index >= max_length_index:
                    # Add the number information to training data
                    data.append(number.copy())

                    # Reset index and number data
                    index = 0
                    number.clear()

    with open(labels_file) as file:

        lines = file.readlines()

        for line in lines:
            number = line[:-1]
            classes.append(number)

    if len(data) != len(classes):
        raise ValueError("Numbers and labels have different size")

    return data, classes


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

training_data, training_classes = extract_data('datasets/digitdata/trainingimages', 'datasets/digitdata/traininglabels')

test_data, test_classes = extract_data('datasets/digitdata/testimages', 'datasets/digitdata/testlabels')

validation_data, validation_classes = extract_data('datasets/digitdata/validationimages', 'datasets/digitdata/validationlabels')
