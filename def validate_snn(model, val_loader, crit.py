def validate_snn(model, val_loader, criterion, device, batch_first=False):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # Reset SNN state if model supports it
            if hasattr(model, "reset"):
                model.reset()

            # forward pass
            spk_rec, _ = model(x, batch_first=batch_first)

            # Compute loss on spike trains
            loss = criterion(spk_rec, y)

            # adding batch loss to total loss
            total_loss += loss.item() * spk_rec.size(1)

            # adding batch correct to total correct (assuming rate encoding)
            num_correct += SF.accuracy_rate(spk_rec, y) * spk_rec.size(1)

            # adding to total number in training set
            total += spk_rec.size(1)

    avg_loss = total_loss / total
    acc = num_correct / total
    return avg_loss, acc