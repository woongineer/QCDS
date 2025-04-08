import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from QuantumNetwork import QNet


def train(q_model, data_loader, optimizer, design):
    q_model.train()
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    for _, (x1, x2, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = q_model(x1, x2, design)
        loss = loss_fn(output, target.double())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def test(q_model, data_loader, design):
    q_model.eval()
    loss_fn = torch.nn.MSELoss()
    epoch_loss = 0
    with torch.no_grad():
        for x1, x2, target in data_loader:
            output = q_model(x1, x2, design)
            instant_loss = loss_fn(output, target.double())
            epoch_loss += instant_loss
    epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def controller_train(q_model, controller, data_loader, controller_optimizer, design, log_prob, entropy, entropy_weight,
                     my_concern):
    controller.train()
    loss_fn = torch.nn.MSELoss()
    q_model.eval()

    total_q_loss = 0.0
    # ---- (1) batch별 q_loss를 합산만 먼저 한다. backward 없음
    for x1, x2, target in data_loader:
        with torch.no_grad():
            q_output = q_model(x1, x2, design)
            q_loss = loss_fn(q_output, target.double())
        total_q_loss += q_loss

    # ---- (2) 전체 loss를 구하고, 한 번만 backward
    # total_q_loss /= len(data_loader.dataset)
    total_q_loss /= len(data_loader)

    if my_concern:
        reward = -total_q_loss
        policy_loss = -log_prob * reward
        entropy_loss = -entropy_weight * entropy
        total_loss = policy_loss + entropy_loss
    else:
        policy_loss = log_prob * total_q_loss
        entropy_loss = -entropy_weight * entropy
        total_loss = policy_loss + entropy_loss

    controller_optimizer.zero_grad()
    total_loss.backward()
    controller_optimizer.step()

    return total_loss.item()



def scheme(controller, train_loader, val_loader, test_loader, controller_optimizer,
           Cepochs, Qepochs, lr, QuantumPATH, entropy_weight, my_concern):
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    best_train_loss, best_val_loss, best_test_loss = 10000, 10000, 10000
    best_train_epoch, best_val_epoch, best_test_epoch = 0, 0, 0
    fidelity_loss_history = []
    best_design = None
    for epoch in range(1, Cepochs + 1):
        print(f"Loop {epoch}")
        controller.eval()
        design, log_prob, entropy = controller()
        q_model = QNet()
        optimizer = optim.Adam(q_model.QuantumLayer.parameters(), lr=lr)

        for q_epoch in range(1, Qepochs + 1):
            fidelity_loss = train(q_model, train_loader, optimizer, design)
            print(f"[Controller Epoch {epoch} | QNet Epoch {q_epoch}] Fidelity loss: {fidelity_loss:.6f}")
            fidelity_loss_history.append(fidelity_loss)

            epoch_train_loss = test(q_model, train_loader, design)
            epoch_test_loss = test(q_model, test_loader, design)
            train_loss_list.append(epoch_train_loss)
            test_loss_list.append(epoch_test_loss)
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_train_epoch = epoch
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_test_epoch = epoch

        epoch_val_loss = controller_train(q_model, controller, val_loader, controller_optimizer, design, log_prob,
                                          entropy, entropy_weight, my_concern)
        val_loss_list.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            best_design = design

        # torch.save({
        #     'epoch': epoch, 'q_model_state_dict': q_model.QuantumLayer.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'controller_optimizer_state_dict': controller_optimizer.state_dict(),
        #     'controller_state_dict': controller.state_dict(),
        #     "test_loss_list": test_loss_list, "val_loss_list": val_loss_list, "train_loss_list": train_loss_list,
        #     "best_val_epoch": best_val_epoch, "best_val_loss": best_val_loss,
        #     "best_train_epoch": best_train_epoch, "best_train_loss": best_train_loss,
        #     "best_test_epoch": best_test_epoch, "best_test_loss": best_test_loss,
        #     "best_design": best_design}, QuantumPATH)

    plot_fidelity_loss(fidelity_loss_history, my_concern, filename="fidelity_loss.png")

    return {"test_loss_list": test_loss_list, "val_loss_list": val_loss_list, "train_loss_list": train_loss_list,
            "best_val_epoch": best_val_epoch, "best_val_loss": best_val_loss,
            "best_train_epoch": best_train_epoch, "best_train_loss": best_train_loss,
            "best_test_epoch": best_test_epoch, "best_test_loss": best_test_loss,
            "best_design": best_design}


def plot_fidelity_loss(fidelity_loss_history, my_concern, filename="fidelity_loss.png"):
    iterations = list(range(1, len(fidelity_loss_history) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fidelity_loss_history, marker='o', linestyle='-')
    plt.xlabel("Global Iteration")
    plt.ylabel("Fidelity Loss (MSE)")
    plt.title("Fidelity Loss")
    plt.grid(True)
    if my_concern:
        filename = filename.replace(".png", "_my_concern.png")
        plt.savefig(filename)
    else:
        plt.savefig(filename)
    plt.close()
