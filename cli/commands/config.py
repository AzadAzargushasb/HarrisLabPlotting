"""
Configuration management commands.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_warning, print_info, print_config_summary
from ..config_loader import load_config, validate_config, create_example_config, ConfigError


@click.group()
def config():
    """
    Configuration file management.

    Commands for creating, validating, and inspecting configuration files.
    """
    pass


@config.command("init")
@click.option("--output", "-o", default="hlplot_config.yaml", type=click.Path(),
              help="Output path for the configuration file.")
@click.option("--force", "-f", is_flag=True,
              help="Overwrite existing file.")
def config_init(output, force):
    """
    Create an example configuration file.

    Generates a well-documented YAML configuration file with all
    available options and sensible defaults.

    \b
    Examples:
      # Create default config
      hlplot config init

      # Create with custom name
      hlplot config init --output my_project.yaml

      # Overwrite existing
      hlplot config init --output config.yaml --force
    """
    output_path = Path(output)

    if output_path.exists() and not force:
        print_error(f"File already exists: {output}")
        print_info("Use --force to overwrite.")
        raise click.Abort()

    try:
        created_path = create_example_config(str(output_path))
        print_success(f"Created configuration file: {created_path}")
        print_info("Edit this file to customize your visualization settings.")

    except Exception as e:
        print_error(f"Error creating configuration file: {e}")
        raise click.Abort()


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True,
              help="Show detailed validation information.")
def config_validate(config_file, verbose):
    """
    Validate a configuration file.

    Checks the configuration file for syntax errors, missing required
    fields, and validates that referenced files exist.

    \b
    Examples:
      # Basic validation
      hlplot config validate my_config.yaml

      # Verbose validation
      hlplot config validate my_config.yaml --verbose
    """
    try:
        print_info(f"Validating configuration file: {config_file}")

        # Try to load
        cfg = load_config(config_file)
        print_success("Configuration file loaded successfully")

        # Validate
        errors = validate_config(cfg)

        if errors:
            console.print()
            print_error(f"Found {len(errors)} validation error(s):")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
            raise click.Abort()
        else:
            print_success("Configuration is valid")

        if verbose:
            print_config_summary(cfg)

    except ConfigError as e:
        print_error(f"Configuration error: {e}")
        raise click.Abort()
    except Exception as e:
        print_error(f"Error validating configuration: {e}")
        raise click.Abort()


@config.command("show")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--section", "-s", default=None,
              help="Show only a specific section (e.g., 'plot', 'camera', 'modularity').")
def config_show(config_file, section):
    """
    Display configuration file contents.

    Shows the parsed configuration with defaults applied.

    \b
    Examples:
      # Show full config
      hlplot config show my_config.yaml

      # Show only plot settings
      hlplot config show my_config.yaml --section plot
    """
    try:
        cfg = load_config(config_file)

        if section:
            if section in cfg:
                console.print()
                console.print(f"[bold]{section}:[/bold]")
                section_data = cfg[section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        console.print(f"  [cyan]{key}:[/cyan] {value}")
                else:
                    console.print(f"  {section_data}")
            else:
                print_error(f"Section '{section}' not found in configuration.")
                print_info(f"Available sections: {', '.join(cfg.keys())}")
        else:
            print_config_summary(cfg)

    except ConfigError as e:
        print_error(f"Configuration error: {e}")
        raise click.Abort()
    except Exception as e:
        print_error(f"Error reading configuration: {e}")
        raise click.Abort()


@config.command("merge")
@click.argument("base_config", type=click.Path(exists=True))
@click.argument("override_config", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output path for merged configuration.")
def config_merge(base_config, override_config, output):
    """
    Merge two configuration files.

    Values from the override config will replace values from the base config.
    Useful for maintaining a base configuration with subject-specific overrides.

    \b
    Examples:
      hlplot config merge base.yaml subject01.yaml --output merged.yaml
    """
    try:
        import yaml

        print_info(f"Loading base config: {base_config}")
        base = load_config(base_config)

        print_info(f"Loading override config: {override_config}")
        override = load_config(override_config)

        # Merge (override takes precedence)
        def deep_merge(base_dict, override_dict):
            result = base_dict.copy()
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(base, override)

        # Save
        output_path = Path(output)
        with open(output_path, 'w') as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

        print_success(f"Merged configuration saved to: {output}")

    except Exception as e:
        print_error(f"Error merging configurations: {e}")
        raise click.Abort()
