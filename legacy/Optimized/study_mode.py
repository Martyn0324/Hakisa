import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# LEARNING PHASE
# She learns from us
# Attention: the first iterations until the print
# will take longer (~11 minutes/10 steps) --> Pytorch memory allocation
# The upcoming ones will be faster (~2 minutes/10 steps)

crossentropy = torch.nn.CrossEntropyLoss()

def train_batch_study(
        states,
        labels,
        chunk,
        chunk_size,
        batch,
        model,
        loss_weights
):

    obs = states[batch].to(device).unsqueeze(0)
    label = labels[(chunk.item()*chunk_size)+batch]

    predicted_actions = model(obs)

    command_type_loss = crossentropy(predicted_actions[0], label[0].to(device))
    action1_loss = crossentropy(predicted_actions[1], label[1].to(device))
    action2_loss = crossentropy(predicted_actions[2], label[2].to(device))

    policy_loss = (command_type_loss * loss_weights[0]) + (action1_loss * loss_weights[1]) + (action2_loss * loss_weights[2])

    policy_loss.backward()

    return policy_loss.item(), command_type_loss.item(), action1_loss.item(), action2_loss.item()

def train_one_epoch_study(
        labels,
        chunks,
        chunk_size,
        batch_size,
        model,
        optimizer,
        human_path,
        loss_weights = [1.0, 1.0, 1.0],
        print_delay = 1000
):

    batches = torch.randperm(chunk_size)
    chunk_order = torch.randperm(chunks)
    steps = 0

    epoch_loss = 0.0

    for chunk in chunk_order:

        states = torch.load(f'{human_path}/human_chunk_states{chunk.item()}.pt')

        if len(states) < chunk_size:

            continue

        for batch in batches:

            batch_loss, ct_loss, act1_loss, act2_loss = train_batch_study(states, labels, chunk, chunk_size, batch, model, loss_weights)

            epoch_loss += batch_loss

            steps += 1

            if steps % batch_size:

                optimizer.step()
                model.zero_grad()

            if steps % print_delay == 0:

                print(f"{steps}")
                print(f"Last Batch Loss: {batch_loss.item()}\tEpoch Loss: {epoch_loss/(steps)}")
                print(f"Command Type Loss: {ct_loss.item()}\tAction1 Loss: {act1_loss.item()}\tAction2 Loss: {act2_loss.item()}")
                print(f"Command Type gradients: {model.pred_command_type.weight.mean()}")
                print(f"Action 1 gradients: {model.pred_action1.weight.mean()}")
                print(f"Action 2 gradients: {model.pred_action2.weight.mean()}")

        del states

    return epoch_loss/steps, steps