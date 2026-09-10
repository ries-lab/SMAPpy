"""Select, review and analyze file-associated regions of interest."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', help='Localization HDF5 files')
    parser.add_argument('--project', help='Open an existing ROI project')
    args = parser.parse_args()
    from ..roi_manager import ROIProject
    from ..roi_manager.gui import ROIManager
    import matplotlib.pyplot as plt
    project = ROIProject.load(args.project) if args.project else ROIProject()
    for path in args.files:
        project.add_file(path)
    manager = ROIManager(project, project_path=args.project)
    plt.show()
    return manager


if __name__ == '__main__':
    main()
