"""Check experiment-local Python dependencies against the published Git tree.

Uses AST parsing, not imports: no Kit startup, checkpoint or dataset is needed.
Framework and third-party dependencies remain the environment's responsibility.
"""
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOTS = {
    'scripts': ROOT/'scripts',
    'isaaclab_tasks': ROOT/'source/isaaclab_tasks/isaaclab_tasks',
    'solo12_symmetry': ROOT/'source/scripts/skrl/solo12_symmetry.py',
}
SEEDS = (
    'scripts.diffusion_policy.experiments.pronk_dagger_v1.run',
    'scripts.diffusion_policy.experiments.pronk_dagger_v1.fit',
    'scripts.diffusion_policy.experiments.pronk_dagger_v1.summarise',
    'scripts.diffusion_policy.experiments.pronk_dagger_v1.dependencies',
    'scripts.diffusion_policy.experiments.pronk_intervention_v1.collect',
    'isaaclab_tasks.direct.solo12.solo12_pronk_v8',
)


def resolve(module):
    head, *tail = module.split('.')
    root = MODULE_ROOTS.get(head)
    if root is None:
        return None
    if root.suffix == '.py':
        return root if not tail else None
    path = root.joinpath(*tail)
    if path.with_suffix('.py').is_file():
        return path.with_suffix('.py')
    if (path/'__init__.py').is_file():
        return path/'__init__.py'
    return None


def source_closure():
    pending = list(SEEDS)
    seen = set()
    files = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = resolve(module)
        if path is None:
            raise FileNotFoundError(f'Missing local dependency: {module}')
        files.add(path)
        # Python executes package initializers as well as the imported module.
        parts = module.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1,len(parts))
                       if resolve('.'.join(parts[:i])) is not None)
        package = module if path.name=='__init__.py' else module.rpartition('.')[0]
        tree = ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                pending.extend(alias.name for alias in node.names if resolve(alias.name) is not None)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parents = package.split('.')
                    prefix = '.'.join(parents[:len(parents)-node.level+1])
                    target = '.'.join(filter(None,(prefix,node.module)))
                    if resolve(target) is None:
                        raise FileNotFoundError(f'{path}: missing relative import {target}')
                else:
                    target = node.module or ''
                if resolve(target) is not None:
                    pending.append(target)
                pending.extend(target+'.'+alias.name for alias in node.names
                               if resolve(target+'.'+alias.name) is not None)
    return sorted(files)


def publication_report(files):
    paths = [p.relative_to(ROOT).as_posix() for p in files]
    tracked = set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT,text=True).splitlines())
    changed = subprocess.check_output(['git','diff','--name-only','HEAD','--',*paths],cwd=ROOT,text=True).splitlines()
    return dict(source_files=len(paths),untracked=sorted(set(paths)-tracked),modified=changed)


if __name__=='__main__':
    report = publication_report(source_closure())
    print(json.dumps(report,indent=2))
    raise SystemExit(1 if report['untracked'] or report['modified'] else 0)
