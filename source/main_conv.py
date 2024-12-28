import source , torch
import torch.nn as nn

def main():
    # Initiliaze dataset
    dataset = source.CelebADataset(num_samples=10000)
    dataset.load()    

    # Datasets and Loaders
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    train_dataset = source.ImageEnhancementDataset(x_train, y_train)
    test_dataset = source.ImageEnhancementDataset(x_test, y_test)
    train_loader = source.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = source.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = source.ImageEnhancementNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    source.train_model(model, train_loader, criterion, optimizer, device, epochs=10)
    metrics = source.evaluate_model(model, test_loader, device, json_path='../results/10000samples_10epochs.json')
    print(metrics)



if __name__ == '__main__':
    main()


