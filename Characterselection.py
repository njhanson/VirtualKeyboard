import numpy as np  # imported for matrix math
#Inputs here list of probablities 1-12. Link to second input in code. Order matrix. Argmax of first fix then second. 

letters = [
        ["A","B","C","D","E","F"],
        ["G","H","I","J","K","L"],
        ["M","N","O","P","Q","R"],
        ["S","T","U","V","W","X"],
        ["Y","Z","1","2","3","4"],
        ["5","6","7","8","9","_"]] 
    #The 6 x 6 grid of all character options. To select for only 8 characters, it would be best to reduce the matrix size (ex: 3 x 3).

def create_flash_matrix(tensor):

    flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.

    tensor = [] #3D array based on the input data from SNNval. A 15x2 for each row, col that is repeated.
    
    for index,rowcol in tensor: #For the 12 flashes
        is_col = index < 6 #First 6 rows are col flashes, next 6 are row flashes.
        is_row = not is_col #The remaining 6 rows are row flashes. 7-12.
        hits = 0
        for spikeobj in rowcol: #15
            for specrowcol in spikeobj:#15
                hits += specrowcol[1] #summing second column of tensor to get total hits for each flash. 
        if is_col:
            column_index = index % 6
            flash_matrix[:, column_index] += hits #summing the hits for each column flash and adding to the corresponding column in the flash matrix.
        else:
            row_index = index % 6
            flash_matrix[row_index, :] += hits #summing the hits for each row flash and adding to the corresponding row in the flash matrix.
    return flash_matrix

#P300 speller cycle character selection function
def p300_speller_cycle(flash_score, repetition=15): # Repetitions is number of flash cycles. Can be changed to increase accuracy (10–15 recommended)

    flash_scores = np.array(flash_score) # Convert the input flash scores to a NumPy array for easier manipulation.

    if flash_scores.shape != (repetition, 12):
        raise ValueError(f"Input must be shape ({repetition}, 12)") #Gives an error if the input flash scores do not match the expected shape of (repetition, 12). This ensures that the function receives the correct format of data for processing.

    # Sum across repetitions
    total_col_scores = np.sum(flash_scores[:, :6], axis=0)
    total_row_scores = np.sum(flash_scores[:, 6:], axis=0)

    # Choose max row and column
    col_idx = np.argmax(total_col_scores)
    row_idx = np.argmax(total_row_scores)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, total_row_scores, total_col_scores

#Printing the letter with the highest score in the row and column.
predicted_letter, row_idx, col_idx, row_scores, col_scores = p300_speller_cycle(flash_scores)

print(f"Row scores: {row_scores}")
print(f"Column scores: {col_scores}")
print(f"Selected Row: {row_idx}, Selected Column: {col_idx}")
print(f"Predicted letter: {predicted_letter}")