def simple_callback(epoch_log, current_epoch, total_epoch):
    train_loss = epoch_log['loss']
    trg_metrics = epoch_log['trg_metrics']
    src_metrics = epoch_log['src_metrics']
    message_head = f'Epoch {current_epoch+1}/{total_epoch}\n'
    message_loss = 'loss: {:<10}\t'.format(train_loss)
    message_src_metrics = ' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in src_metrics.items()])
    message_trg_metrics = ' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in trg_metrics.items()])
    print(message_head + message_loss + message_src_metrics + message_trg_metrics)