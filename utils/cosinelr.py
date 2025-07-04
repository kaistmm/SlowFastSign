import math

def lr_func_cosine(cur_epoch, max_epoch, base_lr, min_lr):
    
	assert min_lr < base_lr
	return (
		min_lr + 
		0.5 * (base_lr - min_lr) * (math.cos(math.pi * cur_epoch / max_epoch) + 1.0)
	)

class CosineAnnealingWarm():
	def __init__(self, optimizer, max_epoch, base_lr, min_lr=0.0, warmup_epochs=0, warmup_start_lr=0.0, **kwargs):
		super(CosineAnnealingWarm, self).__init__()
		self.optimizer = optimizer
		self.max_epoch = max_epoch
		self.warmup_epochs = warmup_epochs
		self.base_lr = base_lr
		self.min_lr = min_lr
		self.warmup_start_lr = warmup_start_lr
		self.warmup_end_lr = lr_func_cosine(warmup_epochs, max_epoch, base_lr, min_lr)
		self.last_epoch = 0
		self._last_lr = warmup_start_lr
	
	def step(self):
		lr = lr_func_cosine(self.last_epoch, self.max_epoch, self.base_lr, self.min_lr)
		if self.last_epoch < self.warmup_epochs:
			alpha = (self.warmup_end_lr - self.warmup_start_lr) / self.warmup_epochs
			lr = self.last_epoch * alpha + self.warmup_start_lr
			self.last_epoch += 1

		self._last_lr = lr

		for param_group in self.optimizer.param_groups:
			if "lr_scale" in param_group:
				param_group["lr"] = lr * param_group["lr_scale"]
			else:
				param_group["lr"] = lr
		return lr
	
	def state_dict(self):
		return {
			'max_epoch': self.max_epoch,
			'base_lr': self.base_lr,
			'min_lr': self.min_lr,
			'warmup_epochs': self.warmup_epochs,
			'warmup_start_lr': self.warmup_start_lr,
			'last_epoch': self.last_epoch,
			'verbose': False,
			'_step_count': self.last_epoch + 1,
			'_get_lr_called_within_step': False,
			'_last_lr': self._last_lr
			}

def Scheduler(optimizer, max_epoch, base_lr, min_lr, warmup_epochs, warmup_start_lr, **kwargs):
	sche_fn = CosineAnnealingWarm(optimizer, max_epoch, base_lr, min_lr, warmup_epochs, warmup_start_lr)
	lr_step = 'epoch'

	print('Initialised CosineAnnealing LR scheduler with warmup')

	return sche_fn, lr_step