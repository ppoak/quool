import importlib
import threading
import multiprocessing
import streamlit as st
from pathlib import Path
from quool.app.tool import task


def setup_task(task_path):
    tasks = Path(task_path).glob('*.py')
    selection = st.sidebar.selectbox("*input task name*", tasks, format_func=lambda x: x.stem)
    if selection is not None:
        spec = importlib.util.spec_from_file_location(selection.stem, selection)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        st.session_state.task = module
    st.sidebar.write(f"**CURRENT TASK: {st.session_state.task.__name__}**")

def get_status(task_path, name):
    status_path = task_path / f"{name}.status"
    if status_path.exists():
        return status_path.read_text()
    return 0

def display_launcher(task_path):
    if st.session_state.get("task") is None:
        st.warning("*no task selected*")
        return
    tasks = filter(
        lambda x: not x.startswith("__") and callable(getattr(st.session_state.task, x)), 
        dir(st.session_state.task)
    )
    name = st.selectbox("*input threading name*", tasks)
    sender = st.text_input("*input sender email*", value=None)
    password = st.text_input("*input password*", value=None, type="password")
    receiver = st.text_input("*input receiver email*", value=None)
    cc = st.text_input("*input cc email*", value=None)
    if name is not None:
        if status:= get_status(task_path, name):
            st.info(f"{name} status: {status}")
            st.session_state.task_name = name
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"run thread anyway", use_container_width=True):
                    t = threading.Thread(target=task(task_path)(getattr(st.session_state.task, name))())
                    t.start()
            with col2:
                if st.button(f"run multiprocessing anyway", use_container_width=True):
                    t = multiprocessing.Process(target=task(task_path)(getattr(st.session_state.task, name))())
                    t.start()
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"start {name} with threading", use_container_width=True):
                    t = threading.Thread(target=task(task_path)(getattr(st.session_state.task, name))())
                    t.start()
            with col2:
                if st.button(f"start {name} with multiprocessing", use_container_width=True):
                    t = multiprocessing.Process(target=task(task_path)(getattr(st.session_state.task, name))())
                    t.start()

def layout(task_path: str | Path = "app/task"):
    st.title("Task")
    task_path = Path(task_path)
    setup_task(task_path)
    display_launcher(task_path)
