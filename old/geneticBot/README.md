# core.py
convert_tasks()

- 根据heuristic和决策树，决定是否convert和convert位置。选择的位置为targetShipyard
- 如果要建新的船厂，将targetShipyard视为已经存在的shipyard，重新计算state[‘closestShipyard’]。这样船只在return的时候会考虑使用这个新的，不存在的船厂，然后到位后自动convert。
- 如果除targetShipyard以外没有船厂，计算targetShipyard最近的ship，并使其强行return

ship_tasks()
如果受到威胁，跑
最后几步，有halite就return，没有halite袭击选定的target

然后根据linear_sum assignment, maximize reward.

spawn_tasks()

- 每一局中process会根据board step, ship number 和 meanHalite预估船的价值。如果>500,造船。
- 如果只剩一个船厂，而且受到威胁，造船
