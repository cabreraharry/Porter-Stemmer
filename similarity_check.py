import csv
import os
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read data from CSV file and return a list of strings
def read_csv(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)  # Store rows as lists
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return data

def calculate_similarity(master_row, slave_row, counter):
    try:       
        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Transform the data into TF-IDF vectors
        tfidf_matrix1 = tfidf_vectorizer.fit_transform([master_row])
        tfidf_matrix2 = tfidf_vectorizer.transform([slave_row])

        # Calculate cosine similarity for rows
        cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

        # Calculate the row difference
        row_difference = abs(len(master_row.split()) - len(slave_row.split()))

        # Adjust the similarity score based on row difference
        adjusted_similarity = cosine_sim[0][0] * 100 - row_difference  # Deduct the row difference

        # Debugging print statement
        # print(f"Similarity {counter}: {adjusted_similarity:.2f}%")
        return adjusted_similarity  # Return the adjusted similarity score
    except Exception as e:
        # Print any exceptions that occur during similarity calculation
        # print(f"Error calculating similarity: {str(e)}")
        return None

def pad_or_truncate(data, target_rows, target_cols):
    padded_data = []
    for i in range(target_rows):
        if i < len(data):
            row = data[i]
            if len(row) < target_cols:
                row += [''] * (target_cols - len(row))
            elif len(row) > target_cols:
                row = row[:target_cols]
            padded_data.append(row)
        else:
            padded_data.append([''] * target_cols)
    return padded_data

if __name__ == "__main__":
    # File paths for the two CSV files
    master_file_path = 'stemmed-dataset_15k-rows_tilaon-antonin.csv'
    slave_file_path =  'output_n.csv'

    # Read data from CSV files
    master_data = read_csv(master_file_path)  # Master file
    slave_data = read_csv(slave_file_path)

    if master_data and slave_data:
        # Determine the number of CPU cores
        num_cores = os.cpu_count()

        with Pool(processes=num_cores) as pool:
            similarities = []

            # Calculate dimensions for padding or truncating slave_data to match master_data
            max_master_rows = len(master_data)
            max_master_cols = max(len(row) for row in master_data)
            max_slave_rows = len(slave_data)
            max_slave_cols = max(len(row) for row in slave_data)

            # Ensure both datasets have the same dimensions
            target_rows = max(max_master_rows, max_slave_rows)
            target_cols = max(max_master_cols, max_slave_cols)

            # Pad or truncate both datasets to match dimensions
            master_data = pad_or_truncate(master_data, target_rows, target_cols)
            slave_data = pad_or_truncate(slave_data, target_rows, target_cols)

            counter = 0

            # Calculate cosine similarity for each cell
            for i in range(target_rows):
                for j in range(target_cols):
                    cell1 = master_data[i][j]
                    cell2 = slave_data[i][j]
                    
                    counter += 1
                    similarity = pool.apply_async(calculate_similarity, args=(cell1, cell2, counter))
                    similarities.append(similarity)

            # Calculate the total similarity
            total_similarity = 0.0
            num_valid_similarities = 0

            for similarity in similarities:
                sim = similarity.get()
                if sim is not None:
                    total_similarity += sim
                    num_valid_similarities += 1

            # Debugging print statements
            print("\nMaster file: ", master_file_path)
            print("Slave file: ", slave_file_path,"\n")
            print("Number of Valid Similarities:", num_valid_similarities)

            # Calculate the average similarity
            average_similarity = total_similarity / num_valid_similarities if num_valid_similarities > 0 else 0.0

            # Format the average similarity as a float with two decimal places
            formatted_average_similarity = f"{average_similarity:.2f}"
            print(f"Average Cell Similarity: {formatted_average_similarity}%\n")
    else:
        print("Data could not be loaded from one or both files.")
