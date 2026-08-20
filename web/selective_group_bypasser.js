import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AIMZ.SelectiveGroupBypasser",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AIMZ_SelectiveGroupBypasser") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.properties = this.properties || {};
                // managedGroups: Array of { id: string|number, title: string, bypassed: boolean }
                this.properties.managedGroups = this.properties.managedGroups || [];

                const node = this;

                // Helper: Get all groups from graph
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

                // Method to toggle bypass state of a group
                node.toggleGroupBypass = function (groupItem) {
                    groupItem.bypassed = !groupItem.bypassed;
                    const targetMode = groupItem.bypassed ? 4 : 0; // 4 = Bypass, 0 = Always

                    const groups = (app.graph && app.graph._groups) ? app.graph._groups : [];
                    const targetGroup = groups.find(g => (g.title || `Group #${g.id}`) === groupItem.title);

                    if (targetGroup) {
                        // Find nodes inside the group bounding box
                        const gMinX = targetGroup.pos[0];
                        const gMinY = targetGroup.pos[1];
                        const gMaxX = gMinX + targetGroup.size[0];
                        const gMaxY = gMinY + targetGroup.size[1];

                        const allNodes = (app.graph && app.graph._nodes) ? app.graph._nodes : [];
                        allNodes.forEach(n => {
                            if (n.id === node.id) return; // Don't bypass self
                            const nCenterX = n.pos[0] + n.size[0] / 2;
                            const nCenterY = n.pos[1] + n.size[1] / 2;

                            if (nCenterX >= gMinX && nCenterX <= gMaxX && nCenterY >= gMinY && nCenterY <= gMaxY) {
                                n.mode = targetMode;
                            }
                        });
                    }

                    node.rebuildDynamicWidgets();
                    if (app.graph) {
                        app.graph.change();
                    }
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Method to remove a group from the manager list
                node.removeManagedGroup = function (index) {
                    node.properties.managedGroups.splice(index, 1);
                    node.rebuildDynamicWidgets();
                };

                // Rebuild dynamic toggle widgets for managed groups only
                node.rebuildDynamicWidgets = function () {
                    // Remove existing dynamic toggle buttons
                    const staticCount = 2; // Combo + Add Button
                    while (node.widgets.length > staticCount) {
                        node.widgets.pop();
                    }

                    // Add dynamic toggle widgets for each managed group
                    node.properties.managedGroups.forEach((gItem, idx) => {
                        const statusIcon = gItem.bypassed ? "🔴 [BYPASS]" : "🟢 [ACTIVE]";
                        const label = `${statusIcon} ${gItem.title}`;

                        // Main Toggle Button
                        node.addWidget("button", label, null, () => {
                            node.toggleGroupBypass(gItem);
                        });

                        // Remove Button
                        node.addWidget("button", `    ↳ ❌ Remove from List`, null, () => {
                            node.removeManagedGroup(idx);
                        });
                    });

                    // Auto-resize node to fit widgets nicely
                    const minWidth = 280;
                    const widgetHeight = 30;
                    const calculatedHeight = 80 + (node.widgets.length * widgetHeight);
                    node.size = [Math.max(node.size[0] || minWidth, minWidth), Math.max(node.size[1] || calculatedHeight, calculatedHeight)];
                    
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Initialize widgets on load
                setTimeout(() => {
                    node.rebuildDynamicWidgets();
                }, 100);

                return r;
            };

            // Restore state when loading workflow
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
