from source.trainer import Trainer

def main():
    params = {'backbone': 'defaultdeep',
              'g_lr': 1e-3,
              'd_lr': 1e-5,}
    #params = {'lr':0.001}
    trainer = Trainer(net_type='gan',
                      num_samples=75000,
                      num_epochs=50,
                      batch_size=64,
                      noise=False,
                      params=params)
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()