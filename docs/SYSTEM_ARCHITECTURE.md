# System Architecture

The Asmodeus architecture is comprised of six modular layers, each responsible for specialized parts of the swarm-native intelligence system. Here’s how the system is structured:

## Layers with Pseudocode

1. **Scout Layer**:
   - Agents tasked with discovering objectives and uncertainties.
   ```python
   class ScoutAgent:
       def discover_tasks(self):
           # Analyze environment -> generate subtask list
           return list_of_tasks
   ```

2. **Worker Layer**:
   - Specialized agents tasked with resolving subtasks assigned by the Coordinator.
   ```python
   class WorkerAgent:
       def execute_task(self, task):
           try:
               result = task.run()
               task.verify(result)
           except Exception as e:
               task.log_failure(e)
           return result
   ```
   - Core function includes retries for transient failure cases.