import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import Dataset
from torchvision import datasets, transforms
import os
import cv2
import json

import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, activation, beta=0.5, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            activation=activation,
            **kwargs,
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.beta = beta
        self.bce_loss_fn = torch.nn.BCELoss()
        # self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE) #, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_iou_history = []
        self.val_iou_history = []
        self.current_epoch_train_loss = None
        self.current_epoch_train_per_image_iou = None
        self.current_epoch_train_dataset_iou = None
        self.current_epoch_val_loss = None
        self.current_epoch_val_per_image_iou = None
        self.current_epoch_val_dataset_iou = None

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image).float()
        mask = mask.float()

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        bce_loss = self.bce_loss_fn(logits_mask, mask)
        dice_loss = self.dice_loss_fn(logits_mask, mask)
        loss = dice_loss + self.beta*bce_loss

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        losses = [x["loss"] for x in outputs]

        avg_loss = torch.stack(losses).mean()

        if stage == "train":
            self.current_epoch_train_loss = avg_loss
            self.current_epoch_train_per_image_iou = per_image_iou
            self.current_epoch_train_dataset_iou = dataset_iou
        elif stage == "valid":
            self.current_epoch_val_loss = avg_loss
            self.current_epoch_val_per_image_iou = per_image_iou
            self.current_epoch_val_dataset_iou = dataset_iou
        metrics = {
            f"{stage}_loss": avg_loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.train_loss_history.append(self.current_epoch_train_loss.cpu().item())
        self.train_iou_history.append(self.current_epoch_train_per_image_iou.cpu().item())
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.val_loss_history.append(self.current_epoch_val_loss.cpu().item())
        self.val_iou_history.append(self.current_epoch_val_per_image_iou.cpu().item())
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return


class PlotMetricsCallback(pl.Callback):
  def __init__(self, save_path):
    super().__init__()
    save_path = os.path.join(save_path, "plots")
    if os.path.exists(save_path):
      raise Exception("remove the plot path !!!")
    os.mkdir(save_path)
    self.save_path = save_path

  def save_text(self, numerical_values, path):
    with open(path, 'w') as f:
      json.dump(numerical_values, f)

  def save_validation_samples(self, val_dataloader, epoch, trainer):
    # یک دسته از داده‌های ارزیابی را بگیرید
    val_samples = next(iter(val_dataloader))
    images, masks = val_samples

    # پیش‌بینی انجام دهید
    with torch.no_grad():
        pred_masks = trainer.model.forward(images.cuda())
        pred_masks = pred_masks > 0.5

    # تبدیل به numpy برای ترسیم
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    pred_masks_np = pred_masks.cpu().numpy()

    # ذخیره 3 نمونه
    save_path = os.path.join(self.save_path, str(epoch))
    os.mkdir(save_path)
    for i in range(min(3, len(images_np))):
      fig, ax = plt.subplots(1, 3, figsize=(15, 5))
      ax[0].imshow(np.transpose(images_np[i], (1, 2, 0)))
      ax[0].set_title("Image")
      ax[1].imshow(masks_np[i][0], cmap="gray")
      ax[1].set_title("Ground Truth")
      ax[2].imshow(pred_masks_np[i][0], cmap="gray")
      ax[2].set_title("Prediction")
      plt.savefig(os.path.join(save_path, f"epoch_{epoch}_sample_{i}.png"))
      plt.close()

  def on_train_epoch_end(self, trainer, pl_module):
    # ترسیم Loss
    plt.figure(figsize=(10, 5))
    plt.title("Loss - Epoch {}".format(trainer.current_epoch))
    plt.plot(pl_module.train_loss_history, label="Train Loss")
    plt.plot(pl_module.val_loss_history, label="Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(self.save_path, f"loss_plot_epoch.png"))
    plt.show()
    plt.close()

    # ترسیم IoU
    plt.figure(figsize=(10, 5))
    plt.title("IoU - Epoch {}".format(trainer.current_epoch))
    plt.plot(pl_module.train_iou_history, label="Train IoU")
    plt.plot(pl_module.val_iou_history, label="Validation IoU")
    plt.legend()
    plt.savefig(os.path.join(self.save_path, f"iou_plot_epoch.png"))
    plt.show()
    plt.close()

    self.save_text(pl_module.train_loss_history, os.path.join(self.save_path, "train_loss.txt"))
    self.save_text(pl_module.val_loss_history, os.path.join(self.save_path, "val_loss.txt"))
    self.save_text(pl_module.train_iou_history, os.path.join(self.save_path, "train_iou.txt"))
    self.save_text(pl_module.val_iou_history, os.path.join(self.save_path, "val_iou.txt"))

    val_dataloader = trainer.val_dataloaders
    self.save_validation_samples(val_dataloader, trainer.current_epoch, trainer)

if __name__ == '__main__':

    # load the data
    x_train_dir = 'datasets/image_dataset_v7.0/segmentation/train/images'
    y_train_dir = 'datasets/image_dataset_v7.0/segmentation/train/labels'
    x_valid_dir = 'datasets/image_dataset_v7.0/segmentation/val/images'
    y_valid_dir = 'datasets/image_dataset_v7.0/segmentation/val/labels'
    x_test_dir = 'datasets/image_dataset_v7.0/segmentation/test/images'
    y_test_dir = 'datasets/image_dataset_v7.0/segmentation/test/labels'

    augmentation = None
    CLASSES = ["signal"]
    max_height = 320           # since height of the images are different (because we use cropping based on lead detection), we have to set a max height to ensuer consistent image shape
    train_dataset = Dataset(images_dir=x_train_dir, masks_dir=y_train_dir, max_height=max_height, augmentation=None, classes=CLASSES,)
    valid_dataset = Dataset(images_dir=x_valid_dir, masks_dir=y_valid_dir, max_height=max_height, augmentation=None, classes=CLASSES,)
    #! if you want to add augmentation, you should use validaton augmentation
    test_dataset = Dataset(images_dir=x_test_dir, masks_dir=y_test_dir, max_height=max_height, augmentation=None, classes=CLASSES,)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)


    # Some training hyperparameters
    EPOCHS = 80
    T_MAX = EPOCHS * len(train_loader)
    # Always include the background as a class
    OUT_CLASSES = len(train_dataset.CLASSES)


    model = CamVidModel("Unet", "resnet34", in_channels=3, out_classes=OUT_CLASSES, activation='sigmoid', beta=0.5)

    save_dir = 'model/segmentation/unet'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename='best-model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='valid_dataset_iou',
            mode='max'
        )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='valid_dataset_iou',
        patience=10,
        mode='max'
    )

    # if os.path.exists("/content/drive/MyDrive/train-log/lightning_logs/plots/"):
    #   os.system("rm -rf /content/drive/MyDrive/train-log/lightning_logs/plots/")

    plot_metrics_callback = PlotMetricsCallback("/content/drive/MyDrive/train-log/lightning_logs/")

    # logger = pl.loggers.TensorBoardLogger(save_dir)
    trainer = pl.Trainer(max_epochs=EPOCHS,
                    default_root_dir=save_dir,
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    callbacks=[checkpoint_callback, early_stop_callback, plot_metrics_callback],
                    #  logger=logger,
                    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        # ckpt_path='/content/drive/MyDrive/ECG-project/code/model/segmentation_unet/lightning_logs/version_15/checkpoints/best-model-epoch=39-val_loss=0.00.ckpt',
        # ckpt_path='/content/drive/MyDrive/ECG-project/code/model/segmentation_unet/model_last.pth',
        # ckpt_path='/content/drive/MyDrive/train-log/lightning_logs/version_4/checkpoints/best-model-epoch=45-val_loss=0.00.ckpt',
    )

    # torch.save(model.state_dict(), "/content/drive/MyDrive/ECG-project/code/model/segmentation_unet/model_40.pth")
    torch.save(model.state_dict(), "/content/drive/MyDrive/train-log/model_last.pth")

    # model.save_pretrained("/content/drive/MyDrive/ECG-project/code/model/segmentation_unet")


    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)


    # visualization
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        model.eval()
        logits = model(images)
    pr_masks = logits

    plt.figure(figsize=(15, 20))
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx < 4:  # Show 4 rows
            plt.subplot(4, 3, idx*3 + 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(4, 3, idx*3 + 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(4, 3, idx*3 + 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
        else:
            break
    plt.tight_layout()
    plt.show()