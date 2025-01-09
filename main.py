from source.trainer import Trainer

def main():
    # params = {'backbone': 'default',
    #          'g_lr': 1e-3,
    #          'd_lr': 1e-4,}
    params = {'lr':0.001}
    trainer = Trainer(net_type='rbf',
                      num_samples=20,
                      num_epochs=1,
                      batch_size=2,
                      noise=True,
                      params=params)
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()