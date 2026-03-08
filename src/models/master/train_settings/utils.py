

def compute_required_steps(batch_size, train_samples, default_early_stop=100):
    # before we used this:
    # early_stop_1K_equivalent = 100  # after gradient steps of 100 epochs * 1000 samples we stop
    # (so with 2K samples we stop after 50 epochs)
    # early_stop = (early_stop_1K_equivalent * 1000) // train_samples
    # epochs = int(epochs * (1000 / train_samples))

    default_batch_size = 32
    default_train_samples = 1000
    default_epochs = 5000

    default_steps_per_epoch = default_train_samples / default_batch_size

    current_run_steps_per_epoch = train_samples / batch_size

    increase = default_steps_per_epoch / current_run_steps_per_epoch

    epochs = int(default_epochs * increase)
    early_stop = int(default_early_stop * increase)

    return epochs, early_stop, increase
