CREATE TABLE `inference_queue_entries` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`image_id` bigint unsigned NOT NULL,
	`idempotency_key` varchar(64) NOT NULL,
	`source_kind` enum('upload','crop') NOT NULL,
	`source_ref` varchar(255) NOT NULL,
	`model_name` varchar(255) NOT NULL,
	`model_version` varchar(32) NOT NULL,
	`run_id` varchar(64) NOT NULL,
	`checkpoint_sha256` varchar(64) NOT NULL,
	`predicted_class` varchar(150) NOT NULL,
	`probability_dog` double NOT NULL,
	`probability_cat` double NOT NULL,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `inference_queue_entries_id` PRIMARY KEY(`id`),
	CONSTRAINT `inference_queue_idempotency_key_unique` UNIQUE(`idempotency_key`),
	CONSTRAINT `inference_queue_image_id_unique` UNIQUE(`image_id`)
);
--> statement-breakpoint
ALTER TABLE `inference_queue_entries` ADD CONSTRAINT `inference_queue_entries_image_id_images_id_fk` FOREIGN KEY (`image_id`) REFERENCES `images`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `inference_queue_created_at_idx` ON `inference_queue_entries` (`created_at`);