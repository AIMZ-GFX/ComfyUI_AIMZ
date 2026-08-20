import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AIMZ.SelectiveGroupBypasser",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AIMZ_SelectiveGroupBypasser") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.properties = this.properties || {};
                // managedGroups: Array of { title: string, bypassed: boolean }
                this.properties.managedGroups = this.properties.managedGroups || [];

                const node = this;
                let draggingIndex = null;

                // Helper: Get all groups from canvas
                function getCanvasGroups() {
                    const groups = (app.graph && app.graph._groups) ? app.graph._groups : [];
                    return groups.map(g => g.title || `Group #${g.id}`);
                }

                // Widget 1: Group Selector Combo
                const groupCombo = node.addWidget("combo", "Group", "", (value) => {}, {
                    values: () => {
                        const all = getCanvasGroups();
                        return all.length > 0 ? all : ["(No Groups Found)"];
                    }
                });

                // Widget 2: Add Group Button
                node.addWidget("button", "➕ Add Group", null, () => {
                    const selected = groupCombo.value;
                    if (!selected || selected === "(No Groups Found)") return;

                    const exists = node.properties.managedGroups.some(g => g.title === selected);
                    if (!exists) {
                        node.properties.managedGroups.push({
                            title: selected,
                            bypassed: false
                        });
                        node.rebuildDynamicWidgets();
                    }
                });

                // Helper to toggle bypass
                node.toggleGroupBypass = function (groupItem) {
                    groupItem.bypassed = !groupItem.bypassed;
                    const targetMode = groupItem.bypassed ? 4 : 0; // 4 = Bypass, 0 = Always

                    const groups = (app.graph && app.graph._groups) ? app.graph._groups : [];
                    const targetGroup = groups.find(g => (g.title || `Group #${g.id}`) === groupItem.title);

                    if (targetGroup) {
                        const gMinX = targetGroup.pos[0];
                        const gMinY = targetGroup.pos[1];
                        const gMaxX = gMinX + targetGroup.size[0];
                        const gMaxY = gMinY + targetGroup.size[1];

                        const allNodes = (app.graph && app.graph._nodes) ? app.graph._nodes : [];
                        allNodes.forEach(n => {
                            if (n.id === node.id) return;
                            const nCenterX = n.pos[0] + n.size[0] / 2;
                            const nCenterY = n.pos[1] + n.size[1] / 2;

                            if (nCenterX >= gMinX && nCenterX <= gMaxX && nCenterY >= gMinY && nCenterY <= gMaxY) {
                                n.mode = targetMode;
                            }
                        });
                    }

                    if (app.graph) app.graph.change();
                    if (app.canvas) app.canvas.setDirty(true, true);
                };

                // Method to remove a group from list
                node.removeManagedGroup = function (index) {
                    node.properties.managedGroups.splice(index, 1);
                    node.rebuildDynamicWidgets();
                };

                // Method to swap / reorder groups
                node.moveManagedGroup = function (fromIndex, toIndex) {
                    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || 
                        fromIndex >= node.properties.managedGroups.length || 
                        toIndex >= node.properties.managedGroups.length) {
                        return;
                    }
                    const item = node.properties.managedGroups.splice(fromIndex, 1)[0];
                    node.properties.managedGroups.splice(toIndex, 0, item);
                    node.rebuildDynamicWidgets();
                };

                // Global mousemove/mouseup listener for smooth drag-reorder
                const onCanvasMouseMove = (e) => {
                    if (draggingIndex === null) return;

                    // Calculate which row index cursor is currently hovering over
                    const nodePos = node.pos;
                    const canvasMouse = app.canvas.graph_to_canvas ? app.canvas.graph_to_canvas(nodePos) : nodePos;
                    // Widget row calculation
                    const staticHeight = 85;
                    const rowHeight = 28;
                    const relativeY = e.canvasY ? (e.canvasY - node.pos[1]) : (e.clientY - node.pos[1]);
                    
                    const targetIdx = Math.floor((relativeY - staticHeight + 20) / rowHeight);
                    if (targetIdx >= 0 && targetIdx < node.properties.managedGroups.length && targetIdx !== draggingIndex) {
                        node.moveManagedGroup(draggingIndex, targetIdx);
                        draggingIndex = targetIdx;
                    }
                };

                const onCanvasMouseUp = () => {
                    if (draggingIndex !== null) {
                        draggingIndex = null;
                        document.removeEventListener("pointermove", onCanvasMouseMove);
                        document.removeEventListener("pointerup", onCanvasMouseUp);
                        document.removeEventListener("mousemove", onCanvasMouseMove);
                        document.removeEventListener("mouseup", onCanvasMouseUp);
                        if (app.canvas) app.canvas.setDirty(true, true);
                    }
                };

                // Rebuild sleek, compact 1-line custom widgets with Drag Handle [≡]
                node.rebuildDynamicWidgets = function () {
                    const staticCount = 2;
                    while (node.widgets.length > staticCount) {
                        node.widgets.pop();
                    }

                    node.properties.managedGroups.forEach((gItem, idx) => {
                        const rowWidget = {
                            type: "aimz_group_row",
                            name: `row_${idx}`,
                            groupItem: gItem,
                            index: idx,
                            draw: function (ctx, node, widget_width, y, widget_height) {
                                const margin = 8;
                                const h = 24;
                                const w = widget_width - margin * 2;
                                const x = margin;

                                ctx.save();

                                // 1. Glassmorphism Row Background
                                ctx.beginPath();
                                ctx.roundRect(x, y, w, h, 4);
                                ctx.fillStyle = gItem.bypassed ? "rgba(45, 20, 20, 0.7)" : "rgba(20, 38, 28, 0.7)";
                                ctx.fill();
                                ctx.strokeStyle = gItem.bypassed ? "rgba(180, 60, 60, 0.4)" : "rgba(60, 160, 90, 0.4)";
                                ctx.lineWidth = 1;
                                ctx.stroke();

                                // 2. Sleek Neon LED Indicator Dot (Left Aligned)
                                const dotX = x + 10;
                                const dotY = y + h / 2;
                                const dotRadius = 4;

                                ctx.beginPath();
                                ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
                                ctx.fillStyle = gItem.bypassed ? "#ff4d4d" : "#00e676";
                                ctx.fill();

                                if (!gItem.bypassed) {
                                    ctx.beginPath();
                                    ctx.arc(dotX, dotY, dotRadius + 2, 0, Math.PI * 2);
                                    ctx.fillStyle = "rgba(0, 230, 118, 0.25)";
                                    ctx.fill();
                                }

                                // 3. Status Tag
                                ctx.font = "600 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
                                ctx.fillStyle = gItem.bypassed ? "#ff7b7b" : "#69f0ae";
                                ctx.textAlign = "left";
                                ctx.textBaseline = "middle";
                                const statusText = gItem.bypassed ? "BYPASS" : "ACTIVE";
                                ctx.fillText(statusText, dotX + 8, dotY);

                                // 4. Group Title
                                const titleX = dotX + 60;
                                ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
                                ctx.fillStyle = gItem.bypassed ? "#8a8a8a" : "#f0f0f0";
                                
                                const maxTitleW = w - 110;
                                let title = gItem.title;
                                if (ctx.measureText(title).width > maxTitleW) {
                                    while (title.length > 3 && ctx.measureText(title + "...").width > maxTitleW) {
                                        title = title.slice(0, -1);
                                    }
                                    title += "...";
                                }
                                ctx.fillText(title, titleX, dotY);

                                // 5. Drag Reorder Handle [≡]
                                const handleW = 18;
                                const handleH = 16;
                                const handleX = x + w - handleW - 24;
                                const handleY = y + (h - handleH) / 2;

                                ctx.beginPath();
                                ctx.roundRect(handleX, handleY, handleW, handleH, 3);
                                ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
                                ctx.fill();

                                // Draw 3 horizontal lines (≡)
                                ctx.strokeStyle = "#888888";
                                ctx.lineWidth = 1.5;
                                for (let i = -3; i <= 3; i += 3) {
                                    ctx.beginPath();
                                    ctx.moveTo(handleX + 4, dotY + i);
                                    ctx.lineTo(handleX + handleW - 4, dotY + i);
                                    ctx.stroke();
                                }

                                // 6. Minimalist Ghost Delete Button [✕]
                                const delBtnW = 16;
                                const delBtnH = 16;
                                const delBtnX = x + w - delBtnW - 4;
                                const delBtnY = y + (h - delBtnH) / 2;

                                ctx.beginPath();
                                ctx.roundRect(delBtnX, delBtnY, delBtnW, delBtnH, 3);
                                ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
                                ctx.fill();

                                ctx.font = "10px sans-serif";
                                ctx.fillStyle = "rgba(255, 100, 100, 0.8)";
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText("✕", delBtnX + delBtnW / 2, delBtnY + delBtnH / 2);

                                ctx.restore();
                            },
                            mouse: function (event, pos, node) {
                                if (event.type === "pointerdown" || event.type === "mousedown") {
                                    const margin = 8;
                                    const h = 24;
                                    const w = node.size[0] - margin * 2;
                                    const x = margin;

                                    const clickX = pos[0];
                                    const clickY = pos[1];

                                    const delBtnW = 16;
                                    const delBtnX = x + w - delBtnW - 4;

                                    const handleW = 18;
                                    const handleX = x + w - handleW - 24;

                                    // 1. Clicked [✕] Delete Button
                                    if (clickX >= delBtnX && clickX <= (delBtnX + delBtnW)) {
                                        node.removeManagedGroup(this.index);
                                        return true;
                                    }

                                    // 2. Clicked [≡] Drag Handle -> Start Drag Reordering
                                    if (clickX >= handleX && clickX <= (handleX + handleW)) {
                                        draggingIndex = this.index;
                                        document.addEventListener("pointermove", onCanvasMouseMove);
                                        document.addEventListener("pointerup", onCanvasMouseUp);
                                        document.addEventListener("mousemove", onCanvasMouseMove);
                                        document.addEventListener("mouseup", onCanvasMouseUp);
                                        return true;
                                    }

                                    // 3. Clicked the Row Body -> Toggle Bypass
                                    if (clickX >= x && clickX < handleX) {
                                        node.toggleGroupBypass(this.groupItem);
                                        return true;
                                    }
                                }
                                return false;
                            },
                            computeSize: function (width) {
                                return [width, 28];
                            }
                        };

                        node.widgets.push(rowWidget);
                    });

                    // Sleek auto-resizing
                    const minWidth = 280;
                    const calculatedHeight = 85 + (node.properties.managedGroups.length * 28);
                    node.size = [Math.max(node.size[0] || minWidth, minWidth), calculatedHeight];

                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };

                setTimeout(() => {
                    node.rebuildDynamicWidgets();
                }, 100);

                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                if (this.rebuildDynamicWidgets) {
                    this.rebuildDynamicWidgets();
                }
                return r;
            };
        }
    }
});
