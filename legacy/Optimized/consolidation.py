import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# CONSOLIDATION PHASE - https://en.wikipedia.org/wiki/Memory_consolidation
# She remembers what she studied, and learns from it.

# In RL, an Epoch consists of an entire episode + training from that episode,
# while a Batch consists of an episode.
# DQN also uses Batch as a number of samples extracted from memory.
# https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
# We'll stick to traditional terminology of ML, to avoid needless confusion.

criterion = torch.nn.MSELoss()

def train_batch_consolidation(
        states,
        next_states,
        actions,
        rewards,
        chunk,
        chunk_size,
        batch,
        model,
        target_network,
        loss_weights,
        gamma
):
    
    obs = states[batch].to(device)
    next_obs = next_states[batch].to(device)
    action = actions[(chunk.item()*chunk_size)+batch]
    reward = rewards[(chunk.item()*chunk_size)+batch].to(device).unsqueeze(0)

    # Compute Q(s_t, a)
    # The model computes Q(s_t), the value of that state
    # Then we select the index of the action that was taken.

    Q_s = model(obs)

    Q_s_a_command_type = Q_s[0][0, action[0].argmax(-1)]
    Q_s_a_action1 = Q_s[1][0, action[1].argmax(-1)]
    Q_s_a_action2 = Q_s[2][0, action[2].argmax(-1)]

    # Compute target Q(s', a'), such that
    # yi = r_t + gamma * Q(s', a'), with the maximum values
    # for each action
    # However, if reached terminal state, yi = r_t
        
    if next_obs is None:

        Q_target = torch.tensor([0.0]*3, device=device)

    else:
        
        Q_target = target_network(next_obs)

        Q_t_a_command_type = Q_target[0][0, Q_target[0].argmax(-1)]
        Q_t_a_action1 = Q_target[1][0, Q_target[1].argmax(-1)]
        Q_t_a_action2 = Q_target[2][0, Q_target[2].argmax(-1)]

    y_i_command_type = reward.item() + gamma * Q_t_a_command_type
    y_i_action1 = reward.item() + gamma * Q_t_a_action1
    y_i_action2 = reward.item() + gamma * Q_t_a_action2

    command_type_loss = criterion(Q_s_a_command_type, y_i_command_type)
    action1_loss = criterion(Q_s_a_action1, y_i_action1)
    action2_loss = criterion(Q_s_a_action2, y_i_action2)

    total_loss = (command_type_loss * loss_weights[0]) + (action1_loss * loss_weights[1]) + (action2_loss * loss_weights[2])

    total_loss.backward()

    return total_loss.item(), command_type_loss.item(), action1_loss.item(), action2_loss.item()

def train_one_epoch_consolidation(
        actions,
        rewards,
        chunks,
        chunk_size,
        batch_size,
        model,
        target_network,
        optimizer,
        human_path,
        loss_weights = [1.0, 1.0, 1.0],
        gamma = 0.995,
        target_delay = 256,
        print_delay = 1000
):

    steps = 0
    epoch_loss = 0.0
    batches = torch.randperm(chunk_size)
    chunk_order = torch.randperm(chunks)

    for chunk in chunk_order:

        states = torch.load(f'{human_path}/human_chunk_states{chunk.item()}.pt')
        next_states = torch.load(f'{human_path}/human_chunk_next_states{chunk.item()}.pt')

        for batch in batches:

            batch_loss, ct_loss, act1_loss, act2_loss = train_batch_consolidation(states, next_states, actions, rewards, chunk, chunk_size, batch, model, target_network, loss_weights, gamma)

            epoch_loss += batch_loss

            steps += 1

            if steps % batch_size:

                optimizer.step()
                model.zero_grad()

            if steps % target_delay:

                target_network.load_state_dict(model.state_dict())

            if steps % print_delay == 0:

                print(f"{steps}")
                print(f"Last Batch Loss: {batch_loss.item()}\tEpoch Loss: {epoch_loss/(steps)}")
                print(f"Command Type Loss: {ct_loss.item()}\tAction1 Loss: {act1_loss.item()}\tAction2 Loss: {act2_loss.item()}")
                print(f"Command Type gradients: {model.pred_command_type.weight.mean()}")
                print(f"Action 1 gradients: {model.pred_action1.weight.mean()}")
                print(f"Action 2 gradients: {model.pred_action2.weight.mean()}")

        del states, next_states

    return epoch_loss/steps, steps