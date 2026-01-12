import React from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ContentBlock } from "@langchain/core/messages";
import { cn } from "@/lib/utils";

interface MultimodalPreviewProps {
    block: ContentBlock;
    removable?: boolean;
    onRemove?: () => void;
    size?: "sm" | "md" | "lg";
    className?: string;
}

const sizeClasses = {
    sm: "h-8 w-8",
    md: "h-12 w-12",
    lg: "h-16 w-16",
};

export const MultimodalPreview: React.FC<MultimodalPreviewProps> = ({
    block,
    removable = false,
    onRemove,
    size = "md",
    className,
}) => {
    const isImage = block.type === "image";
    const isFile = block.type === "file";

    const name =
        (isImage &&
            "metadata" in block &&
            block.metadata &&
            typeof block.metadata === "object" &&
            "name" in block.metadata &&
            typeof block.metadata.name === "string"
            ? block.metadata.name
            : null) ||
        (isFile &&
            "metadata" in block &&
            block.metadata &&
            typeof block.metadata === "object" &&
            "filename" in block.metadata &&
            typeof block.metadata.filename === "string"
            ? block.metadata.filename
            : null) ||
        "File";

    return (
        <div
            className={cn(
                "relative group flex items-center gap-2 rounded-lg border bg-muted p-2",
                className,
            )}
        >
            {isImage && "data" in block && "mime_type" in block ? (
                <img
                    src={`data:${block.mime_type};base64,${block.data}`}
                    alt={name}
                    className={cn("rounded object-cover", sizeClasses[size])}
                />
            ) : (
                <div
                    className={cn(
                        "flex items-center justify-center rounded bg-gray-200 text-xs",
                        sizeClasses[size],
                    )}
                >
                    PDF
                </div>
            )}
            <span className="max-w-[150px] truncate text-sm">{name}</span>
            {removable && onRemove && (
                <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
                    onClick={onRemove}
                >
                    <X className="h-4 w-4" />
                </Button>
            )}
        </div>
    );
};