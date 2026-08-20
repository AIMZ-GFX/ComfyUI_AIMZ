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

                // Rebuild sleek, compact 1-line custom widgets with clean Left-Alignment & Modern LED Dot UI
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

                                // 1. Modern Glassmorphism Row Background
                                ctx.beginPath();
                                ctx.roundRect(x, y, w, h, 4);
                                ctx.fillStyle = gItem.bypassed ? "rgba(45, 20, 20, 0.7)" : "rgba(20, 38, 28, 0.7)";
                                ctx.fill();
                                ctx.strokeStyle = gItem.bypassed ? "rgba(180, 60, 60, 0.4)" : "rgba(60, 160, 90, 0.4)";
                                ctx.lineWidth = 1;
                                ctx.stroke();

                                // 2. Sleek Neon LED Indicator Dot (Left Aligned)
                                const dotX = x + 12;
                                const dotY = y + h / 2;
                                const dotRadius = 4;

                                ctx.beginPath();
                                ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
                                ctx.fillStyle = gItem.bypassed ? "#ff4d4d" : "#00e676";
                                ctx.fill();

                                // Glow effect for Active LED
                                if (!gItem.bypassed) {
                                    ctx.beginPath();
                                    ctx.arc(dotX, dotY, dotRadius + 2, 0, Math.PI * 2);
                                    ctx.fillStyle = "rgba(0, 230, 118, 0.25)";
                                    ctx.fill();
                                }

                                // 3. Status Tag (Left Aligned, Modern Typography)
                                ctx.font = "600 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
                                ctx.fillStyle = gItem.bypassed ? "#ff7b7b" : "#69f0ae";
                                ctx.textAlign = "left";
                                ctx.textBaseline = "middle";
                                const statusText = gItem.bypassed ? "BYPASS" : "ACTIVE";
                                ctx.fillText(statusText, dotX + 8, dotY);

                                // 4. Group Title (Left Aligned, Clean)
                                const titleX = dotX + 62;
                                ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
                                ctx.fillStyle = gItem.bypassed ? "#8a8a8a" : "#f0f0f0";
                                
                                const maxTitleW = w - 90;
                                let title = gItem.title;
                                if (ctx.measureText(title).width > maxTitleW) {
                                    while (title.length > 3 && ctx.measureText(title + "...").width > maxTitleW) {
                                        title = title.slice(0, -1);
                                    }
                                    title += "...";
                                }
                                ctx.fillText(title, titleX, dotY);

                                // 5. Minimalist Ghost Delete Button (Right Aligned)
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

                                    // Check if clicked the [X] button
                                    if (clickX >= delBtnX && clickX <= (delBtnX + delBtnW)) {
                                        node.removeManagedGroup(this.index);
                                        return true;
                                    }

                                    // Otherwise clicked the row -> Toggle Bypass
                                    if (clickX >= x && clickX < delBtnX) {
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
                    const minWidth = 260;
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
