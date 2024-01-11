import matplotlib.pyplot as plt
import torch

def visualize_images(chunk_number, objective_number):

    states = torch.load(f'{HUMAN_DATA_PATH}/human_chunk_states{chunk_number}.pt')

    for i in range(len(states)):

        state = states[i] # (Channel, Height, Width)
        state = state.permute(1, 2, 0).numpy()
        
        plt.imshow(state)
        plt.title(f"Chunk Number: {chunk_number}\tObjective Number: {objective_number}\nCurrent Item: {i}")
        plt.show()

    return None

print("\nWhat is the data path?")

HUMAN_DATA_PATH = input()

print("\nInsert Chunk Number and, after that, which objetive shall be visualized (1, 2 or 3)")

chunk_number = int(input())
objective_number = int(input())

print("\nStand by...\n")

visualize_images(chunk_number, objective_number)