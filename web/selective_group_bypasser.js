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
                const groupCombo = node.addWidget("combo", "Select Group", "", (value) => {}, {
                    values: () => {
                        const all = getCanvasGroups();
                        return all.length > 0 ? all : ["(No Groups Found)"];
                    }
                });

                // Widget 2: Add Group Button
                node.addWidget("button", "➕ Add Selected Group", null, () => {
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

                // Rebuild sleek, compact 1-line custom widgets
                node.rebuildDynamicWidgets = function () {
                    // Retain only the first 2 static widgets (Combo & Add button)
                    const staticCount = 2;
                    while (node.widgets.length > staticCount) {
                        node.widgets.pop();
                    }

                    // Create a single compact row widget per group
                    node.properties.managedGroups.forEach((gItem, idx) => {
                        const rowWidget = {
                            type: "aimz_group_row",
                            name: `row_${idx}`,
                            groupItem: gItem,
                            index: idx,
                            draw: function (ctx, node, widget_width, y, widget_height) {
                                const margin = 10;
                                const h = 26;
                                const w = widget_width - margin * 2;
                                const x = margin;

                                // Row Background Box
                                ctx.save();
                                ctx.beginPath();
                                ctx.roundRect(x, y, w, h, 4);
                                ctx.fillStyle = gItem.bypassed ? "#3a1e1e" : "#1e3324";
                                ctx.fill();
                                ctx.strokeStyle = gItem.bypassed ? "#7a2a2a" : "#2a6a3b";
                                ctx.lineWidth = 1;
                                ctx.stroke();

                                // Status Badge (Left Pill)
                                const badgeW = 68;
                                const badgeH = 18;
                                const badgeX = x + 4;
                                const badgeY = y + 4;
                                ctx.beginPath();
                                ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3);
                                ctx.fillStyle = gItem.bypassed ? "#a83232" : "#2e8b57";
                                ctx.fill();

                                ctx.font = "bold 10px sans-serif";
                                ctx.fillStyle = "#ffffff";
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText(gItem.bypassed ? "BYPASS" : "ACTIVE", badgeX + badgeW / 2, badgeY + badgeH / 2);

                                // Group Title Text
                                ctx.font = "12px sans-serif";
                                ctx.fillStyle = gItem.bypassed ? "#cccccc" : "#ffffff";
                                ctx.textAlign = "left";
                                ctx.textBaseline = "middle";
                                const textX = badgeX + badgeW + 8;
                                const maxTextW = w - badgeW - 35;
                                
                                let title = gItem.title;
                                if (ctx.measureText(title).width > maxTextW) {
                                    while (title.length > 3 && ctx.measureText(title + "...").width > maxTextW) {
                                        title = title.slice(0, -1);
                                    }
                                    title += "...";
                                }
                                ctx.fillText(title, textX, y + h / 2);

                                // Delete (X) Button on the right
                                const delBtnW = 20;
                                const delBtnH = 18;
                                const delBtnX = x + w - delBtnW - 4;
                                const delBtnY = y + 4;

                                ctx.beginPath();
                                ctx.roundRect(delBtnX, delBtnY, delBtnW, delBtnH, 3);
                                ctx.fillStyle = "#444444";
                                ctx.fill();

                                ctx.font = "bold 11px sans-serif";
                                ctx.fillStyle = "#ff6666";
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText("✕", delBtnX + delBtnW / 2, delBtnY + delBtnH / 2);

                                ctx.restore();
                            },
                            mouse: function (event, pos, node) {
                                if (event.type === "pointerdown" || event.type === "mousedown") {
                                    const margin = 10;
                                    const h = 26;
                                    const w = node.size[0] - margin * 2;
                                    const x = margin;
                                    const y = this.last_y || 0;

                                    const clickX = pos[0];
                                    const clickY = pos[1];

                                    const delBtnW = 20;
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
                                return [width, 30];
                            }
                        };

                        node.widgets.push(rowWidget);
                    });

                    // Compact auto-resizing
                    const minWidth = 280;
                    const calculatedHeight = 90 + (node.properties.managedGroups.length * 32);
                    node.size = [Math.max(node.size[0] || minWidth, minWidth), calculatedHeight];

                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Initialize on load
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
