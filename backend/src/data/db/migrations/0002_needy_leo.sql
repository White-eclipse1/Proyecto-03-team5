-- Se eliminan temporalmente las FKs para poder cambiar los IDs a UNSIGNED.
ALTER TABLE `annotations`
DROP FOREIGN KEY `annotations_image_id_images_id_fk`;
--> statement-breakpoint

ALTER TABLE `annotations`
DROP FOREIGN KEY `annotations_category_id_categories_id_fk`;
--> statement-breakpoint


-- IDs y llaves foráneas pasan a UNSIGNED.
ALTER TABLE `images`
MODIFY COLUMN `id` bigint unsigned AUTO_INCREMENT NOT NULL;
--> statement-breakpoint

ALTER TABLE `categories`
MODIFY COLUMN `id` bigint unsigned AUTO_INCREMENT NOT NULL;
--> statement-breakpoint

ALTER TABLE `annotations`
MODIFY COLUMN `id` bigint unsigned AUTO_INCREMENT NOT NULL;
--> statement-breakpoint

ALTER TABLE `annotations`
MODIFY COLUMN `image_id` bigint unsigned NOT NULL;
--> statement-breakpoint

ALTER TABLE `annotations`
MODIFY COLUMN `category_id` bigint unsigned NOT NULL;
--> statement-breakpoint


-- Campos de anotaciones.
ALTER TABLE `annotations`
MODIFY COLUMN `area` double NOT NULL;
--> statement-breakpoint

ALTER TABLE `annotations`
ADD `updated_at` timestamp DEFAULT (now()) NOT NULL
ON UPDATE CURRENT_TIMESTAMP;
--> statement-breakpoint


-- Categorías.
ALTER TABLE `categories`
ADD `color` varchar(7) NOT NULL;
--> statement-breakpoint


-- Metadatos de imágenes para MinIO y validaciones.
ALTER TABLE `images`
MODIFY COLUMN `storage_key` varchar(512) NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
MODIFY COLUMN `mime_type` varchar(100) NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
MODIFY COLUMN `width` int unsigned NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
MODIFY COLUMN `height` int unsigned NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
MODIFY COLUMN `size_bytes` bigint unsigned NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
ADD `status` enum('pending','in_progress','completed')
DEFAULT 'pending' NOT NULL;
--> statement-breakpoint

ALTER TABLE `images`
ADD `updated_at` timestamp DEFAULT (now()) NOT NULL
ON UPDATE CURRENT_TIMESTAMP;
--> statement-breakpoint


-- Evita que dos registros apunten al mismo objeto en MinIO.
ALTER TABLE `images`
ADD CONSTRAINT `images_storage_key_unique`
UNIQUE(`storage_key`);
--> statement-breakpoint


-- Índices para búsquedas y filtros.
CREATE INDEX `annotations_image_category_idx`
ON `annotations` (`image_id`,`category_id`);
--> statement-breakpoint

CREATE INDEX `images_status_idx`
ON `images` (`status`);
--> statement-breakpoint

CREATE INDEX `images_created_at_idx`
ON `images` (`created_at`);
--> statement-breakpoint

CREATE INDEX `images_status_created_at_idx`
ON `images` (`status`,`created_at`);
--> statement-breakpoint


-- Se restauran las llaves foráneas después de cambiar los tipos.
ALTER TABLE `annotations`
ADD CONSTRAINT `annotations_image_id_images_id_fk`
FOREIGN KEY (`image_id`)
REFERENCES `images`(`id`)
ON DELETE cascade
ON UPDATE no action;
--> statement-breakpoint

ALTER TABLE `annotations`
ADD CONSTRAINT `annotations_category_id_categories_id_fk`
FOREIGN KEY (`category_id`)
REFERENCES `categories`(`id`)
ON DELETE restrict
ON UPDATE no action;