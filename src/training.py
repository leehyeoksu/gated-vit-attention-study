import os
import gc
import torch
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm

from .evaluating import evaluate
from .re_make_Gate import compute_gate_loss


def train(
    model,
    dataloader,
    test_loader,
    epochs,
    criterion,
    scaler,
    device,
    optimizer,
    experiment_name,
    path,
    patience=3,
    min_delta=0.001,
    gate_lambda=0.01,
):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(experiment_name)

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with mlflow.start_run():
        model = model.to(device)

        best_acc = 0.0
        counter = 0

        # 파라미터 로깅
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", dataloader.batch_size)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("lr", optimizer.param_groups[0]["lr"])
        mlflow.log_param("patience", patience)
        mlflow.log_param("min_delta", min_delta)
        mlflow.log_param("gate_lambda", gate_lambda)

        use_amp = (device == "cuda")

        for i in tqdm(range(epochs), desc=experiment_name):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            epoch_gate_loss = 0.0

            # =====================
            # Train
            # =====================
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)

                # forward + cls loss
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    out = model(x)
                    cls_loss = criterion(out, y)

                # gate loss
                gate_loss = compute_gate_loss(model).to(device)
                loss = cls_loss + gate_lambda * gate_loss

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                bn = x.size(0)
                total += bn
                total_loss += loss.item() * bn
                correct += (out.argmax(dim=1) == y).sum().item()
                epoch_gate_loss += gate_loss.item()

            train_loss = total_loss / total
            train_acc = correct / total

            # =====================
            # Test
            # =====================
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # =====================
            # MLflow Logging
            # =====================
            avg_gate_loss = epoch_gate_loss / max(len(dataloader), 1)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)
            mlflow.log_metric("gate_loss", avg_gate_loss, step=i)

            print(
                f"Epoch {i:03d} | "
                f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
                f"Test Loss={test_loss:.4f} | Test Acc={test_acc:.4f} | "
                f"Gate Loss={avg_gate_loss:.5f}"
            )

            # =====================
            # Early Stopping
            # =====================
            if test_acc > best_acc + min_delta:
                best_acc = test_acc
                counter = 0
                torch.save(model.state_dict(), path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early Stopping Triggered at epoch {i}")
                    mlflow.log_metric("stopped_epoch", i)
                    break

        # =====================
        # Best model 로드 후 저장
        # =====================
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            mlflow.log_artifact(path)
            mlflow.pytorch.log_model(model, artifact_path="best_model")
            print("최고 성능 모델이 MLflow에 저장되었습니다.")
        else:
            print("저장된 모델 가중치가 없습니다. (성능 개선 없음)")

        mlflow.log_metric("best_test_accuracy", best_acc)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()