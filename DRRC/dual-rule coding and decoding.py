####################
from numpy import fromfile, uint8
import numpy as np



######Chunking the input binary and randomizing the first position
def segment(ascii_binary):
    ascii_length = len(ascii_binary)
    pad = (5 - ascii_length % 5) % 5
    temp = ascii_binary + '0' * pad
    n = (ascii_length + pad) // 5
    temp = np.array(list(temp)).reshape((n, 5))
    temp_first_bit = (temp[:, 0])
    temp_first_bit_bool = [bool(int(x)) for x in temp_first_bit]
    c = temp_first_bit_bool
    ratio = 0.0
    numl = 0
    ###### Balancing the ratio of first position sequences 0 and 1 so that the final coding yields a balanced base sequence GC
    while not 0.40 <= ratio <= 0.60:
        numl += 1
        length1 = len(temp_first_bit_bool)
        # X_1 iteration
        x_1 = 0.1 + numl * 5 * 10 ** -5
        u = 3.9
        y = np.zeros(length1)
        x_n_1 = x_1
        for k in range(length1):
            if x_n_1 > 0.5:
                y[k] = 1
            else:
                y[k] = 0
            x_n_2 = u * (1 - x_n_1) * x_n_1
            x_n_1 = x_n_2
        y = y.astype(bool)
        c = np.logical_xor(temp_first_bit_bool, y)
        c = (c).astype(int)
        n = len(c)
        ratio = np.count_nonzero(c == 0) / n

    #
    #print(''.join(y.astype(int).astype(str)))
    #########Output the initial value of the chaotic sequence for subsequent decoding
    print("Initial value of chaotic sequence: ")
    print(x_1)
    #print("The binary sequence after the dissimilarity is: ")
    #print(c)
    print("The weight of 0 in the binary sequence after the dissimilarity is: " + str(ratio))
    c = ''.join(c.astype(int).astype(str))
    c = np.reshape(list(c), (n, 1))
    temp[:, 0] = c[:, 0]
    preprocessed_binary_data = np.reshape(temp, n * 5)
    print("The length of the binary sequence to be encoded is: ")
    print(len(preprocessed_binary_data))
    #print("The binary order to be encoded is: ")
    #print(preprocessed_binary_data)
    return preprocessed_binary_data,x_1,numl

def encode_sequence(sequence):
    rule_0 = {
        '0000': 'ACT',
        '0001': 'AGT',
        '0010': 'ATC',
        '0011': 'ATG',
        '0100': 'TCA',
        '0101': 'TGA',
        '0110': 'TAC',
        '0111': 'TAG',
        '1100': 'CAT',
        '1101': 'CTA',
        '1110': 'CAA',
        '1111': 'CTT',
        '1000': 'GTA',
        '1001': 'GAT',
        '1010': 'GTT',
        '1011': 'GAA'
    }
    rule_1 = {
        '0000': 'ACG',
        '0001': 'AGC',
        '0010': 'AGG',
        '0011': 'ACC',
        '0100': 'TGC',
        '0101': 'TCG',
        '0110': 'TCC',
        '0111': 'TGG',
        '1100': 'CAC',
        '1101': 'CAG',
        '1110': 'CTC',
        '1111': 'CTG',
        '1000': 'GAG',
        '1001': 'GAC',
        '1010': 'GTG',
        '1011': 'GTC'
    }

    first_bit = int(sequence[0])
    key = sequence[1:]

    if first_bit == 0:
        if key in rule_0:
            return rule_0[key]
    elif first_bit == 1:
        if key in rule_1:
            return rule_1[key]

    return "Invalid sequence"


def decode_sequence(seq):
    rule_0 = {
        'ACT': '00000',
        'AGT': '00001',
        'ATC': '00010',
        'ATG': '00011',
        'TCA': '00100',
        'TGA': '00101',
        'TAC': '00110',
        'TAG': '00111',
        'CAT': '01100',
        'CTA': '01101',
        'CAA': '01110',
        'CTT': '01111',
        'GTA': '01000',
        'GAT': '01001',
        'GTT': '01010',
        'GAA': '01011'
    }
    rule_1 = {
        'ACG': '10000',
        'AGC': '10001',
        'AGG': '10010',
        'ACC': '10011',
        'TGC': '10100',
        'TCG': '10101',
        'TCC': '10110',
        'TGG': '10111',
        'CAC': '11100',
        'CAG': '11101',
        'CTC': '11110',
        'CTG': '11111',
        'GAG': '11000',
        'GAC': '11001',
        'GTG': '11010',
        'GTC': '11011'
    }
    #########Counting the number of GCs within a base slice for decoding purposes
    count_gc = seq.count('G') + seq.count('C')
    if count_gc == 1:
        if seq in rule_0:
            return rule_0[seq]
    elif count_gc == 2:
        if seq in rule_1:
            return rule_1[seq]
    return "11111"

#############Converting files to binary
def read_bits_from_file(path, segment_length=168, need_logs=True):

    if need_logs:
        print("Read binary matrix from file: " + path)

    matrix, values = [], fromfile(file=path, dtype=uint8)
    for current, value in enumerate(values):
        matrix += list(map(int, list(str(bin(value))[2:].zfill(8))))

    if len(matrix) % segment_length != 0:
        matrix += [0] * (segment_length - len(matrix) % segment_length)

    if need_logs:
        print("There are " + str(len(matrix)) + " segments of length " + str(segment_length) + " in the inputted file.")

    return matrix


##########Convert binary to file
def write_bits_to_file(bits, bits_size, file_path):
    # Intercept the first n digits
    bits = bits[:bits_size]

    # Converting one-dimensional arrays to Numpy arrays
    bit_array = np.array(bits)

    # Reshape the Numpy array to the right size
    reshaped_array = bit_array.reshape(-1, 8)

    # Convert each byte from binary to integer
    byte_array = [int(''.join(map(str, byte_row)), 2) for byte_row in reshaped_array]

    # Write byte data to file
    with open(file_path, 'wb') as file:
        file.write(bytes(byte_array))
def calculate_GC_content(input_str):
    total_length = len(input_str)
    count_GC = input_str.count('G') + input_str.count('C')
    GC_content = count_GC / total_length * 100
    return GC_content


######Path to the input file
binary=read_bits_from_file('D:\\destop\\DRRC\\data\\pictures\\United Nations Flag.bmp', need_logs=True)

binary_sequence = ''.join(map(str, binary))
temp=segment(binary_sequence)
print(temp[2])
temp = temp[0]
temp = ''.join(temp)
#Turn temp into an array of n rows and 5 columns.
n = len(temp) // 5
# Slicing a string and assembling it into a list of n rows and 5 columns
substrings = [temp[i*5:(i+1)*5].zfill(5) for i in range(n)]
a=len(substrings)
output = ""  # Creating an Empty String
for i in range(len(substrings)):
    DNA_sequence= encode_sequence(substrings[i])
    output += DNA_sequence
#print(output)
DNAsequence=output
print("The logical storage density is",len(binary)/len(DNAsequence))
GC_content = calculate_GC_content(DNAsequence)
print("GC content: {:.2f}%".format(GC_content))
print("Coding Completed")

################Decoding of DNA Sequence.
n = len(DNAsequence) // 3
# Slicing a string and assembling it into a list of n rows and 5 columns
DNAstrings = [DNAsequence[i*3:(i+1)*3].zfill(3) for i in range(n)]
Binary_output = ""
for i in range(len(DNAstrings)):
    Binary_sequence= decode_sequence(DNAstrings[i])
    Binary_output += Binary_sequence
#print(Binary_output)

source_first_bit = ''
for i in range(0, len(Binary_output), 5):
    source_first_bit += Binary_output[i]

#print(source_first_bit)
source_first_bit = np.fromstring(source_first_bit, dtype=np.uint8) - ord('0')
source_first_bit = np.logical_xor(source_first_bit.astype(bool), False)
print(source_first_bit.shape)
x_1 = float(input('Please enter the initial value of the chaotic sequence: '))
u = 3.9
y = np.zeros(len(source_first_bit))
x_n_1 = x_1

for k in range(len(source_first_bit)):
    if x_n_1 > 0.5:
        y[k] = 1
    else:
        y[k] = 0

    x_n_2 = u * (1 - x_n_1) * x_n_1
    x_n_1 = x_n_2
y = y.astype(bool)
temp_first_bit = np.logical_xor(source_first_bit, y)
temp_first_bit = (temp_first_bit).astype(int)
temp_first_bit = ''.join(temp_first_bit.astype(int).astype(str))
temp_first_bit = np.reshape(list(temp_first_bit), (len(temp_first_bit), 1))
Binary_output = np.array(list(Binary_output)).reshape((len(Binary_output)//5), 5)
Binary_output[:, 0] = temp_first_bit[:, 0]
primary_binary_data = np.reshape(Binary_output, len(Binary_output) * 5)
primary_binary_list = [int(bit) for bit in primary_binary_data]

############Recover the base sequence to the original file and save it in the corresponding path, here 3840480bit is the size of the file to be recovered.
decoded_file=write_bits_to_file( primary_binary_list, 3840480, 'D:\\destop\\DRRC\\generate file\\target.bmp')
print("Decoding completed" )