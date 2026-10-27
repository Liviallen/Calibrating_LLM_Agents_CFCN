"""Per-task progress snapshots and terminal updates, including redirected jobs."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def _write_terminal(message):
    # Shell redirection captures stdout, but /dev/tty remains the user's terminal.
    if not sys.stdout.isatty():
        try:
            with open('/dev/tty', 'w', encoding='utf-8') as terminal:
                terminal.write(message + '\n')
                terminal.flush()
        except OSError:
            pass  # Noninteractive jobs still have their log and JSON snapshot.


class Progress:
    def __init__(self, task, total, completed=0):
        self.task = task
        self.total = total
        self.completed = completed
        self.started = time.monotonic()
        self.last_print = self.started
        self.last_count = completed
        self.run = Path(os.environ['TLV_RUN_DIR'])
        self.folder = self.run / 'progress'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f'{task}.json'
        self.every = max(1, int(os.environ.get('TLV_PROGRESS_EVERY', '10')))
        self.state = 'running'
        self._publish(force=True)

    def _publish(self, force=False, detail=''):
        now = time.monotonic()
        elapsed = now - self.started
        data = dict(task=self.task, completed=self.completed, total=self.total,
                    percent=100 * self.completed / self.total if self.total else 0,
                    state=self.state, elapsed_seconds=round(elapsed, 1),
                    pid=os.getpid(), updated_at=time.time(), detail=detail)
        # Each task owns its own file; replacement avoids partially read snapshots.
        with tempfile.NamedTemporaryFile(mode='w', dir=self.folder, delete=False) as tmp:
            json.dump(data, tmp)
            temporary = tmp.name
        os.replace(temporary, self.path)
        if force or self.completed - self.last_count >= self.every or now - self.last_print >= 30:
            combination, mode = self.run.parts[-2:]
            message = (f'[{combination}/{mode} | {self.task}] '
                       f'{self.completed}/{self.total} ({data["percent"]:.1f}%) '
                       f'{self.state}; elapsed {elapsed:.0f}s')
            if detail:
                message += f'; {detail}'
            print(message, flush=True)
            _write_terminal(message)
            self.last_count = self.completed
            self.last_print = now

    def update(self, completed):
        self.completed = completed
        self._publish()

    def finish(self):
        self.completed = self.total
        self.state = 'completed'
        self._publish(force=True)

    def fail(self, error):
        self.state = 'failed'
        # Avoid writing arbitrary exception text that could contain request credentials.
        self._publish(force=True, detail=type(error).__name__ + ': see task log')

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is None:
            self.finish()
        else:
            self.fail(error)
        return False
