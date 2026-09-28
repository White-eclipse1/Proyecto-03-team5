CREATE TABLE `annotations` (
	`id` bigint AUTO_INCREMENT NOT NULL,
	`image_id` bigint NOT NULL,
	`category_id` bigint NOT NULL,
	`bbox_x` double NOT NULL,
	`bbox_y` double NOT NULL,
	`bbox_width` double NOT NULL,
	`bbox_height` double NOT NULL,
	`area` double,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `annotations_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `categories` (
	`id` bigint AUTO_INCREMENT NOT NULL,
	`name` varchar(150) NOT NULL,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `categories_id` PRIMARY KEY(`id`),
	CONSTRAINT `categories_name_unique` UNIQUE(`name`)
);
--> statement-breakpoint
CREATE TABLE `images` (
	`id` bigint AUTO_INCREMENT NOT NULL,
	`filename` varchar(255) NOT NULL,
	`storage_key` varchar(512),
	`mime_type` varchar(100),
	`width` int,
	`height` int,
	`size_bytes` bigint,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `images_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `annotations` ADD CONSTRAINT `annotations_image_id_images_id_fk` FOREIGN KEY (`image_id`) REFERENCES `images`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE `annotations` ADD CONSTRAINT `annotations_category_id_categories_id_fk` FOREIGN KEY (`category_id`) REFERENCES `categories`(`id`) ON DELETE restrict ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `annotations_image_id_idx` ON `annotations` (`image_id`);--> statement-breakpoint
CREATE INDEX `annotations_category_id_idx` ON `annotations` (`category_id`);